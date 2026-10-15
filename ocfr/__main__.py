import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


class Perception(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.bottom = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(32, nodes + 1, 1)

    def forward(self, x):
        first = self.first(x)
        bottom = self.bottom(F.avg_pool2d(first, 2))
        feature = self.up(torch.cat([first, F.interpolate(bottom, first.shape[-2:], mode='bilinear', align_corners=False)], 1))
        return self.head(feature), feature


class Dynamics(nn.Module):
    def __init__(self, command_dim, c):
        super().__init__()
        self.c = c
        self.edge = nn.Sequential(nn.Linear(83, 64), nn.ReLU(), nn.Linear(64, 64))
        self.gru = nn.GRUCell(103 + command_dim, 128)
        self.mean = nn.Linear(128, c['horizon'] * 39)
        self.variance = nn.Linear(128, c['horizon'] * 39)
        self.renderer = nn.Sequential(nn.Conv2d(35, 32, 1), nn.ReLU(), nn.Conv2d(32, 4, 1))

    def forward(self, state, command, hidden):
        n = len(state)
        first = state[:, None].expand(n, n, -1)
        second = state[None].expand(n, n, -1)
        delta = first[:, :, :2] - second[:, :, :2]
        distance = delta.norm(dim=-1, keepdim=True)
        overlap = torch.exp(-distance)
        relation = torch.cat([delta, distance, overlap, (distance < .1).float()], -1)
        message = self.edge(torch.cat([first, second, relation], -1))
        message = (message * (1 - torch.eye(n, device=state.device))[:, :, None]).sum(1)
        if self.c['variant'] == 'no-action':
            command = command * 0
        hidden = self.gru(torch.cat([state, message, command[None].expand(n, -1)], 1), hidden)
        mean = self.mean(hidden).reshape(n, self.c['horizon'], 39).transpose(0, 1)
        logvar = self.variance(hidden).reshape(n, self.c['horizon'], 39).transpose(0, 1).clamp(-6, 4)
        return mean, logvar, hidden

    def render(self, states, size):
        n = len(states)
        y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing='ij')
        grid = torch.stack([x, y])[None].expand(n, -1, -1, -1)
        local = (grid - states[:, :2, None, None]) / states[:, 2:4].exp().clamp_min(.01)[:, :, None, None]
        appearance = states[:, 6:38, None, None].expand(-1, -1, size, size)
        confidence = states[:, 38:39, None, None].expand(-1, -1, size, size)
        output = self.renderer(torch.cat([appearance, local, confidence], 1))
        weights = output[:, 3].softmax(0)
        return (weights[:, None] * output[:, :3].sigmoid()).sum(0)


def load_episode(path, size):
    d = np.load(path, allow_pickle=False)
    rgb = torch.tensor(d['rgb']).permute(0, 3, 1, 2).float() / 255
    rgb = F.interpolate(rgb, (size, size), mode='bilinear', align_corners=False)
    return d, rgb


def track(perception, rgb, c):
    states, masks = [], []
    previous = None
    previous_mask = None
    height, width = rgb.shape[-2:]
    with torch.no_grad():
        for image in rgb:
            logits, feature = perception(image[None])
            probability = logits[0].softmax(0)
            labels = probability.argmax(0)
            current, binary = [], []
            for node in range(c['nodes']):
                mask = labels == node + 1
                binary.append(mask)
                if mask.any():
                    yy, xx = torch.where(mask)
                    center = torch.stack([xx.float().mean() / width, yy.float().mean() / height])
                    box = torch.tensor([(xx.max() - xx.min() + 1) / width, (yy.max() - yy.min() + 1) / height]).clamp_min(1e-5).log()
                    appearance = feature[0, :, mask].mean(1)
                    confidence = probability[node + 1][mask].mean()
                else:
                    center, box, appearance, confidence = torch.zeros(2), torch.full((2,), -5.), torch.zeros(32), torch.tensor(0.)
                current.append(torch.cat([center, box, torch.zeros(2), appearance, confidence[None]]))
            current, binary = torch.stack(current), torch.stack(binary)
            if previous is not None:
                centroid = torch.cdist(current[:, :2], previous[:, :2], p=1)
                appearance = 1 - F.normalize(current[:, 6:38], dim=1) @ F.normalize(previous[:, 6:38], dim=1).T
                intersection = (binary[:, None] & previous_mask[None]).sum((-1, -2))
                union = (binary[:, None] | previous_mask[None]).sum((-1, -2)).clamp_min(1)
                cost = centroid + .5 * appearance + 1 - intersection / union
                cost += (1 - torch.eye(c['nodes'])) * c['identity_penalty']
                rows, columns = linear_sum_assignment(cost.numpy())
                ordering = rows[np.argsort(columns)]
                current, binary = current[ordering], binary[ordering]
                current[:, 4:6] = current[:, :2] - previous[:, :2]
            states.append(current)
            masks.append(binary)
            previous, previous_mask = current, binary
    return torch.stack(states), torch.stack(masks)


def prepare(c, out, source):
    source = Path(source)
    frame = pd.read_csv(source)
    if not {'episode', 'split', 'task', 'failure', 'terminal', 'cause'} <= set(frame):
        raise ValueError('Manifest requires episode,split,task,failure,terminal,cause')
    frame.episode = frame.episode.map(lambda p: str((source.parent / p).resolve()))
    if frame.episode.duplicated().any() or frame.loc[frame.split == 'train', 'failure'].any():
        raise ValueError('Use distinct episodes and successful-only predictor training')
    frame.to_csv(out / 'manifest.csv', index=False)


def train(c, out):
    torch.manual_seed(c['seed'])
    frame = pd.read_csv(out / 'manifest.csv')
    fitting = frame[frame.split == 'train']
    perception = Perception(c['nodes'])
    optimizer = torch.optim.AdamW(perception.parameters(), lr=c['learning_rate'])
    for _ in range(c['perception_epochs']):
        for row in fitting.itertuples():
            d, rgb = load_episode(row.episode, c['size'])
            masks = F.interpolate(torch.tensor(d['masks'])[:, None].float(), (c['size'], c['size']), mode='nearest')[:, 0].long()
            for ids in torch.randperm(len(rgb)).split(c['batch_size']):
                loss = F.cross_entropy(perception(rgb[ids])[0], masks[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    perception.eval()
    torch.save(perception.state_dict(), out / 'perception.pt')
    states, episodes = [], []
    for row in fitting.itertuples():
        d, rgb = load_episode(row.episode, c['size'])
        state, _ = track(perception, rgb, c)
        states.append(state)
        episodes.append((state, torch.tensor(d['command'], dtype=torch.float32), rgb))
    pooled = torch.cat(states)
    mean, scale = pooled.mean((0, 1)), pooled.std((0, 1)).clamp_min(.01)
    np.savez(out / 'normalization.npz', mean=mean.numpy(), scale=scale.numpy())
    model = Dynamics(episodes[0][1].shape[-1], c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    for _ in range(c['epochs']):
        for raw_state, command, rgb in episodes:
            state = (raw_state - mean) / scale
            hidden = torch.zeros(c['nodes'], 128)
            for t in range(len(state) - c['horizon']):
                prediction, logvar, hidden = model(state[t], command[t], hidden.detach())
                target = state[t + 1:t + 1 + c['horizon']]
                weights = torch.linspace(1, 1.5, c['horizon'])[:, None, None]
                loss = (.5 * ((target - prediction).square() * (-logvar).exp() + logvar) * weights).mean()
                relation = prediction[:, :, None, :2] - prediction[:, None, :, :2]
                true_relation = target[:, :, None, :2] - target[:, None, :, :2]
                loss += .1 * F.mse_loss(relation, true_relation)
                horizon = c['render_horizon'] - 1
                rendered = model.render(prediction[horizon] * scale + mean, c['size'])
                loss += .2 * F.l1_loss(rendered, rgb[t + horizon + 1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    torch.save(model.state_dict(), out / 'dynamics.pt')
    paths = [float(command[:, :3].norm(dim=1).sum()) for _, command, _ in episodes]
    (out / 'path_scale.json').write_text(json.dumps(dict(path_scale=float(np.median(paths)))))


def episode_scores(c, out, episode):
    d, rgb = load_episode(episode, c['size'])
    perception = Perception(c['nodes'])
    perception.load_state_dict(torch.load(out / 'perception.pt', weights_only=True, map_location='cpu'))
    perception.eval()
    start = time.perf_counter()
    raw, _ = track(perception, rgb, c)
    perception_ms = (time.perf_counter() - start) * 1000 / len(rgb)
    normalization = np.load(out / 'normalization.npz')
    mean, scale = torch.tensor(normalization['mean']), torch.tensor(normalization['scale'])
    state = (raw - mean) / scale
    command = torch.tensor(d['command'], dtype=torch.float32)
    model = Dynamics(command.shape[-1], c)
    model.load_state_dict(torch.load(out / 'dynamics.pt', weights_only=True, map_location='cpu'))
    model.eval()
    hidden = torch.zeros(c['nodes'], 128)
    archive, output, pixels = {}, [], []
    horizons = [1] if c['variant'] == 'single-horizon' else [1, 4, 8, 12]
    with torch.no_grad():
        for t in range(len(state)):
            available = [h for h in horizons if t - h in archive and h <= c['horizon']]
            score = torch.zeros(c['nodes'])
            pixel = np.zeros((c['size'], c['size']))
            for h in available:
                forecast, logvar = archive[t - h]
                predicted, variance = forecast[h - 1], logvar[h - 1].exp()
                if c['variant'] == 'no-variance':
                    variance = torch.ones_like(variance)
                score += ((state[t] - predicted).square() / variance).sum(1)
                if c['variant'] != 'no-relation':
                    for other in [0, 2]:
                        observed = state[t, :, :2] - state[t, other, :2]
                        expected = predicted[:, :2] - predicted[other, :2]
                        relation_var = variance[:, :2] + variance[other, :2]
                        score += c['relation_weight'] * ((observed - expected).square() / relation_var.clamp_min(1e-6)).sum(1)
            if available:
                score /= len(available)
            if t - 1 in archive:
                predicted = archive[t - 1][0][0] * scale + mean
                rendered = model.render(predicted, c['size'])
                pixel = (rendered - rgb[t]).abs().mean(0).numpy()
            prediction, logvar, hidden = model(state[t], command[t], hidden)
            archive[t] = prediction, logvar
            archive.pop(t - c['horizon'] - 1, None)
            output.append(score.numpy())
            pixels.append(pixel)
    path_scale = json.loads((out / 'path_scale.json').read_text(encoding='utf-8'))['path_scale']
    progress = np.minimum(1, command[:, :3].norm(dim=1).cumsum(0).numpy() / max(path_scale, 1e-6))
    return np.array(output), progress, np.array(pixels), perception_ms


def normalized_features(scores, progress, location, scale, c):
    bins = np.minimum(c['progress_bins'] - 1, (progress * c['progress_bins']).astype(int))
    z = (scores - location[bins]) / scale[bins]
    if c['variant'] == 'no-phase':
        z = scores
    filtered, features = 0., []
    for value in z.max(1):
        new = c['ewma'] * filtered + (1 - c['ewma']) * value
        features.append([new, new - filtered])
        filtered = new
    return np.array(features), z.argmax(1)


def calibrate(c, out):
    frame = pd.read_csv(out / 'manifest.csv')
    cache = [(row, *episode_scores(c, out, row.episode)) for row in frame[frame.split == 'val'].itertuples()]
    location, scale = [], []
    normal_scores = np.concatenate([scores for row, scores, _, _, _ in cache if not row.failure])
    normal_progress = np.concatenate([progress for row, _, progress, _, _ in cache if not row.failure])
    bins = np.minimum(c['progress_bins'] - 1, (normal_progress * c['progress_bins']).astype(int))
    for index in range(c['progress_bins']):
        values = normal_scores[bins == index]
        if not len(values):
            values = normal_scores
        median = np.median(values, 0)
        location.append(median)
        scale.append(np.maximum(np.median(abs(values - median), 0), 1e-3))
    location, scale = np.array(location), np.array(scale)
    features, targets, normal = [], [], []
    for row, scores, progress, _, _ in cache:
        f, _ = normalized_features(scores, progress, location, scale, c)
        terminal = min(row.terminal, len(f)) if row.failure else len(f)
        target = np.exp(-(terminal - np.arange(terminal)) / c['target_horizon']) if row.failure else np.zeros(terminal)
        features.append(f[:terminal])
        targets.append(target)
        normal.append(np.full(terminal, not row.failure))
    x = np.concatenate(features)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    x = torch.tensor((x - mean) / std, dtype=torch.float32)
    y = torch.tensor(np.concatenate(targets), dtype=torch.float32)
    is_normal = torch.tensor(np.concatenate(normal))
    calibrator = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=.01)
    for _ in range(c['calibration_steps']):
        logits = calibrator(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y) + c['normal_penalty'] * logits[is_normal].sigmoid().square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    coefficient = calibrator.weight.detach().numpy()[0]
    intercept = float(calibrator.bias.detach()[0])
    peaks = []
    for row, scores, progress, _, _ in cache:
        if row.failure:
            continue
        f, _ = normalized_features(scores, progress, location, scale, c)
        p = 1 / (1 + np.exp(-np.clip(((f - mean) / std) @ coefficient + intercept, -700, 700)))
        peaks.append(max([min(p[t:t + c['persistence']]) for t in range(len(p) - c['persistence'] + 1)], default=0))
    np.savez(out / 'calibration.npz', location=location, scale=scale, mean=mean, std=std, coefficient=coefficient,
             intercept=intercept, threshold=np.quantile(peaks, .95))


def run(c, out):
    frame = pd.read_csv(out / 'manifest.csv')
    cal = np.load(out / 'calibration.npz')
    rows = []
    for index, row in enumerate(frame[frame.split == 'test'].itertuples()):
        start = time.perf_counter()
        scores, progress, pixels, perception_ms = episode_scores(c, out, row.episode)
        features, object_id = normalized_features(scores, progress, cal['location'], cal['scale'], c)
        logits = ((features - cal['mean']) / cal['std']) @ cal['coefficient'] + cal['intercept']
        p = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        count = 0
        for t, probability in enumerate(p):
            count = count + 1 if probability >= cal['threshold'] else 0
            alert = count == c['persistence']
            rows.append(dict(episode=index, path=row.episode, task=row.task, frame=t, risk=probability,
                             alert=alert, object_id=int(object_id[t]), score=features[t, 0],
                             perception_ms=perception_ms))
        np.savez_compressed(out / f'pixels_{index}.npz', residual=pixels)
        latency = 1000 * (time.perf_counter() - start) / len(scores)
        for result in rows[-len(scores):]:
            result['latency_ms'] = latency
    pd.DataFrame(rows).to_csv(out / 'predictions.csv', index=False)


def evaluate(c, out):
    manifest = pd.read_csv(out / 'manifest.csv').set_index('episode')
    d = pd.read_csv(out / 'predictions.csv')
    episodes, leads, attributions, dominance = [], [], [], []
    false_alarms, minutes = 0, 0.
    for index, g in d.groupby('episode'):
        truth = manifest.loc[g.path.iloc[0]]
        terminal = min(int(truth.terminal), len(g)) if truth.failure else len(g)
        before = g[g.frame < terminal]
        alerts = before[before.alert]
        detected = len(alerts) > 0
        episodes.append(dict(failure=truth.failure, score=before.risk.max(), detected=detected))
        if truth.failure and detected:
            leads.append((terminal - alerts.frame.iloc[0]) / c['fps'])
            attributions.append(int(alerts.object_id.iloc[0]) == int(truth.cause))
        if not truth.failure:
            false_alarms += len(alerts)
            minutes += len(g) / c['fps'] / 60
        raw = np.load(g.path.iloc[0])
        if truth.failure and 'masks' in raw.files:
            masks = F.interpolate(torch.tensor(raw['masks'])[:, None].float(), (c['size'], c['size']), mode='nearest')[:, 0].numpy() == truth.cause + 1
            residual = np.load(out / f'pixels_{index}.npz')['residual']
            inside = (residual * masks).sum((1, 2))
            outside = (residual * ~masks).sum((1, 2))
            dominance.extend((outside / (inside + 1e-8)).tolist())
    e = pd.DataFrame(episodes)
    frame_target = []
    for row in d.itertuples():
        truth = manifest.loc[row.path]
        frame_target.append(float(np.exp(-(truth.terminal - row.frame) / c['target_horizon'])) if truth.failure and row.frame < truth.terminal else 0.)
    frame_target = np.array(frame_target)
    ece = sum(len(ids) / len(d) * abs(frame_target[ids].mean() - d.risk.iloc[ids].mean()) for ids in np.array_split(np.argsort(d.risk), min(15, len(d))))
    result = dict(method=c['variant'], event_auroc=roc_auc_score(e.failure, e.score), event_f1=f1_score(e.failure, e.detected),
                  median_lead_seconds=float(np.median(leads)) if leads else np.nan, false_alarms_per_minute=false_alarms / minutes,
                  object_attribution_accuracy=float(np.mean(attributions)) if attributions else np.nan,
                  brier_soft_target=float(np.mean((d.risk.to_numpy() - frame_target) ** 2)), ece=ece,
                  background_dominance=float(np.mean(dominance)) if dominance else np.nan,
                  latency_ms=d.latency_ms.mean())
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'predictions.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    for episode, g in d.groupby('episode'):
        ax.plot(g.frame, g.risk, alpha=.4)
    ax.set(xlabel='Causal frame', ylabel='Failure risk')
    fig.tight_layout()
    fig.savefig(out / 'warnings.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'calibrate', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'no-action', 'no-relation', 'no-phase', 'no-variance', 'single-horizon']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
