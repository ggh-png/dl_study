"""MNIST VAE의 한 변수씩 순차 튜닝. python examples/vae_tuning.py --help"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
# CLI는 화면 없이 저장하고, 노트북 import 시에는 기존 inline backend를 유지함.
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class Config:
    # 단계마다 이 설정 중 한 항목만 바꾸며, 각 실험은 처음부터 학습함.
    beta: float = 1.0
    latent_dim: int = 2
    learning_rate: float = 1e-3
    hidden_dim: int = 400
    batch_size: int = 128
    anneal_fraction: float = 0.0
    likelihood: str = 'bce'
    gaussian_sigma: float = 0.1


def load_model_class():
    # 학습 노트북의 클래스 정의만 읽어 동일한 ResNet을 재사용함; 데이터/학습 셀은 실행하지 않음.
    notebook = ROOT / 'notebooks/02_vae_mnist.ipynb'
    document = json.loads(notebook.read_text())
    source = next(''.join(c['source']) for c in document['cells']
                  if c['cell_type'] == 'code' and 'class VAE(' in ''.join(c['source']))
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), 'exec'), namespace)
    return namespace['VAE'], hashlib.sha256(source.encode()).hexdigest()


def seed_all(seed):
    # 실험마다 초기 난수 상태를 초기화함; 다른 모델 크기 간 가중치가 동일하다는 뜻은 아님.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_data(data_dir, train_limit, val_limit, seed):
    # 테스트 데이터는 탐색 단계에서 읽지 않음. 원본 학습 집합을 중복 없이 54,000/6,000으로 분할함.
    dataset = MNIST(data_dir, train=True, download=True)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    validation_ids, training_ids = order[:6000], order[6000:]
    if train_limit:
        training_ids = training_ids[:train_limit]
    if val_limit:
        validation_ids = validation_ids[:val_limit]
    def tensors(ids):
        # MNIST 원본 uint8을 한 번만 float로 변환하여 반복 실험의 이미지 전처리 비용을 줄임.
        return TensorDataset(dataset.data[ids].unsqueeze(1).float() / 255.0)
    return tensors(training_ids), tensors(validation_ids), training_ids, validation_ids


def beta_at(config, epoch, epochs):
    # warmup 첫 epoch는 0, 지정된 마지막 warmup epoch부터 목표 beta에 도달함.
    warmup = max(2, math.ceil(epochs * config.anneal_fraction))
    if config.anneal_fraction == 0:
        return config.beta
    return config.beta * min(1.0, (epoch - 1) / (warmup - 1))


def loss_parts(logits, images, mu, logvar, config):
    # BCE는 픽셀별 합, KL은 잠재 차원별 합을 낸 뒤 배치 평균을 사용함.
    bce = F.binary_cross_entropy_with_logits(logits, images, reduction='none').flatten(1).sum(1).mean()
    pixel_mse = (logits.sigmoid() - images).square().flatten(1).mean(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    if config.likelihood == 'bce':
        reconstruction = bce
    else:
        # 고정 분산 Gaussian의 음의 로그 우도: SSE/(2 sigma²) + 픽셀 수 * log(sigma sqrt(2pi)).
        # sigma는 별도 고정 설정임; BCE와 이 NLL의 숫자를 직접 비교하지 않음.
        pixels = images[0].numel()
        reconstruction = pixels * (pixel_mse / (2 * config.gaussian_sigma ** 2)
                                  + math.log(config.gaussian_sigma * math.sqrt(2 * math.pi)))
    return reconstruction, kl, bce, pixel_mse


def run_epoch(model, loader, config, beta, optimizer=None, eval_seed=1042):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(6)
    count = 0
    # 검증용 epsilon을 매번 동일하게 생성하여 epoch별 샘플링 변동을 줄임.
    # 검증 배치 크기도 128로 고정하고, 학습 난수 스트림을 소모하지 않음.
    generator = torch.Generator(device=DEVICE).manual_seed(eval_seed)
    with torch.set_grad_enabled(training):
        for (images,) in loader:
            images = images.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
                logits, mu, logvar = model(images)
            else:
                mu, logvar = model.encode(images)
                epsilon = torch.randn(mu.shape, device=DEVICE, generator=generator)
                logits = model.decode(mu + (0.5 * logvar).exp() * epsilon)
            reconstruction, kl, bce, mse = loss_parts(logits, images, mu, logvar, config)
            total = reconstruction + beta * kl
            # 실험 전체에 같은 BCE+KL 기준을 사용함. beta 변경으로 점수의 정의가 바뀌지 않음.
            reference = bce + kl
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite loss')
            if training:
                total.backward()
                optimizer.step()
            values = [total, reconstruction, kl, reference, bce, mse]
            totals += np.array([v.item() for v in values]) * len(images)
            count += len(images)
    return dict(zip(('total', 'reconstruction', 'kl', 'reference', 'bce', 'pixel_mse'),
                    (totals / count).tolist()))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruction_image(model, data, path):
    # 모든 실험에서 동일한 검증 이미지 10장을 평균 z=mu로 복원하여 비교함.
    images = data.tensors[0][:10].to(DEVICE)
    model.eval()
    with torch.inference_mode():
        restored = model(images, sample=False)[0].sigmoid().cpu()
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for j in range(10):
        for i, image in enumerate((images[j].cpu(), restored[j])):
            axes[i, j].imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
    axes[0, 0].set_title('Original')
    axes[1, 0].set_title('mu reconstruction')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train_trial(model_class, config, training, validation, epochs, seed, path, stage):
    path.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    model = model_class(config.latent_dim, config.hidden_dim).to(DEVICE)
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history, best_score, best_state = [], float('inf'), None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        beta = beta_at(config, epoch, epochs)
        train = run_epoch(model, train_loader, config, beta, optimizer)
        val = run_epoch(model, val_loader, config, beta)
        record = {'epoch': epoch, 'effective_beta': beta}
        record.update({f'train_{k}': v for k, v in train.items()})
        record.update({f'val_{k}': v for k, v in val.items()})
        history.append(record)
        # annealing 중간의 모델이 아니라 목표 beta에 도달한 모델들 중에서 선택함.
        if beta == config.beta and val['reference'] < best_score:
            best_score, best_record = val['reference'], record.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'{stage} {path.name} epoch={epoch}/{epochs} beta={beta:.3g} '
              f"val_recon={val['reconstruction']:.3f} kl={val['kl']:.3f} ref={val['reference']:.3f}", flush=True)
    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'config': asdict(config), 'best_metrics': best_record,
                'history': history, 'seed': seed}, path / 'best.pt')
    write_csv(path / 'history.csv', history)
    reconstruction_image(model, validation, path / 'reconstruction.png')
    # 각각의 지표를 분리한 train/validation 곡선을 저장함.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, metric in zip(axes, ('total', 'reconstruction', 'kl')):
        for split in ('train', 'val'):
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history], label=split)
        ax.set(title=metric, xlabel='epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'curves.png', dpi=120)
    plt.close(fig)
    result = {'trial': path.name, 'stage': stage, **asdict(config), **best_record,
              'parameters': sum(p.numel() for p in model.parameters()),
              'seconds': time.perf_counter() - start,
              'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2**20 if DEVICE.type == 'cuda' else 0,
              'reconstruction_image': str(path / 'reconstruction.png')}
    (path / 'result.json').write_text(json.dumps(result, indent=2))
    del model, optimizer
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def run_search(output, data_dir=Path('/data'), epochs=10, train_limit=6000, val_limit=1000, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_class, source_hash = load_model_class()
    training, validation, train_ids, val_ids = prepare_data(data_dir, train_limit, val_limit, seed)
    np.savez(output / 'split_indices.npz', train=train_ids.numpy(), validation=val_ids.numpy())
    manifest = {'epochs': epochs, 'train_size': len(training), 'val_size': len(validation),
                'seed': seed, 'model_source_sha256': source_hash, 'torch': str(torch.__version__),
                'device': str(DEVICE), 'selection': 'validation BCE + KL with fixed weight 1',
                'likelihood': 'BCE soft-target MNIST; Gaussian supported, not mixed into search'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    current, rows, cache, selections = Config(), [], {}, []
    stages = [('beta', [0.1, 0.5, 1.0, 2.0]), ('latent_dim', [8, 16, 32, 64]),
              ('learning_rate', [1e-3, 3e-4, 1e-4]), ('hidden_dim', [400, 800]),
              ('batch_size', [32, 64, 128]), ('anneal_fraction', [0.0, 0.5])]
    # 기존 기본 모델도 첫 단계의 대조군으로 포함됨. 각 단계는 그 직전 선택값을 고정함.
    for parameter, candidates in stages:
        configs = list(dict.fromkeys([current] + [replace(current, **{parameter: v}) for v in candidates]))
        contenders = []
        for config in configs:
            assert sum(getattr(current, k) != getattr(config, k) for k in asdict(current)) <= 1
            key = json.dumps(asdict(config), sort_keys=True)
            if key not in cache:
                path = output / f'{len(rows):02d}_{parameter}_{getattr(config, parameter)}'
                result = train_trial(model_class, config, training, validation, epochs, seed, path, parameter)
                rows.append(result)
                cache[key] = result
                write_csv(output / 'summary.csv', rows)
            contenders.append((config, cache[key]))
        current, chosen = min(contenders, key=lambda item: item[1]['val_reference'])
        selections.append({'stage': parameter, 'selected_trial': chosen['trial'],
                           'candidates': [r['trial'] for _, r in contenders], 'config': asdict(current)})
        (output / 'selections.json').write_text(json.dumps(selections, indent=2))
        print(f'SELECTED {parameter}: {asdict(current)}', flush=True)
    winner = cache[json.dumps(asdict(current), sort_keys=True)]
    (output / 'winner.json').write_text(json.dumps(winner, indent=2))
    return rows, selections, winner


def confirm_search(search_output, output, data_dir=Path('/data'), epochs=20, seed=42):
    """전체 데이터에서 baseline/탐색 선택값을 확인하고 검증으로 선택한 모델만 최종 테스트함."""
    output, search_output = Path(output), Path(search_output)
    output.mkdir(parents=True, exist_ok=False)
    winner = json.loads((search_output / 'winner.json').read_text())
    config = Config(**{k: winner[k] for k in Config.__dataclass_fields__})
    model_class, source_hash = load_model_class()
    search_manifest = json.loads((search_output / 'manifest.json').read_text())
    if source_hash != search_manifest['model_source_sha256']:
        raise ValueError('탐색 이후 모델 정의가 바뀌었습니다. 동일 모델로 다시 탐색하세요.')
    training, validation, train_ids, val_ids = prepare_data(data_dir, 0, 0, seed)
    np.savez(output / 'split_indices.npz', train=train_ids.numpy(), validation=val_ids.numpy())
    rows = []
    # 확인 단계는 탐색 경로의 출발점과 최종점을 비교함; 추가적인 하이퍼파라미터 탐색은 하지 않음.
    for label, candidate in [('baseline', Config()), ('tuned', config)]:
        rows.append(train_trial(model_class, candidate, training, validation, epochs, seed,
                                output / label, 'full_confirmation'))
        write_csv(output / 'summary.csv', rows)
    selected = min(rows, key=lambda row: row['val_reference'])
    selected_config = Config(**{k: selected[k] for k in Config.__dataclass_fields__})
    checkpoint = torch.load(output / selected['trial'] / 'best.pt', map_location='cpu', weights_only=True)
    model = model_class(selected_config.latent_dim, selected_config.hidden_dim).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    # 테스트는 이 시점에 처음 로드하며, 테스트 점수로 후보를 다시 고르지 않음.
    test = MNIST(data_dir, train=False, download=True)
    test_data = TensorDataset(test.data.unsqueeze(1).float() / 255.0)
    metrics = run_epoch(model, DataLoader(test_data, batch_size=128), selected_config, selected_config.beta)
    checkpoint['test_metrics'] = metrics
    torch.save(checkpoint, output / 'selected.pt')
    final = {'selected_trial': selected['trial'], 'config': asdict(selected_config),
             'test_metrics': metrics, 'epochs': epochs, 'train_size': len(training),
             'val_size': len(validation), 'seed': seed, 'model_source_sha256': source_hash}
    (output / 'final.json').write_text(json.dumps(final, indent=2))
    print('FINAL ' + json.dumps(final), flush=True)
    return rows, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirm-from', type=Path, help='완료된 탐색 경로: 전체 데이터로 baseline과 선택값 확인')
    parser.add_argument('--data-dir', type=Path, default=Path('/data'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train-limit', type=int, default=6000)
    parser.add_argument('--val-limit', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 2 or args.train_limit < 0 or args.val_limit < 0:
        parser.error('epochs >= 2; limits >= 0 (0 means all)')
    torch.set_num_threads(2)
    if args.confirm_from:
        confirm_search(args.confirm_from, args.output, args.data_dir, args.epochs, args.seed)
    else:
        run_search(args.output, args.data_dir, args.epochs, args.train_limit, args.val_limit, args.seed)


if __name__ == '__main__':
    main()
