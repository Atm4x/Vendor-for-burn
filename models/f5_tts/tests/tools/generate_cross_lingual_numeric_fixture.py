"""Generate numerical references from the Cross-Lingual F5-TTS Python formulas."""

import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'fixtures'
RNG = random.Random(0xF5201009)
RATE = 24000
HOP = 256


def predicted_frames(reference_samples, units, predicted_speed, user_speed):
    local_speed = 0.5 if units < 4 else user_speed
    generated_seconds = units / predicted_speed / local_speed
    return reference_samples // HOP + int(generated_seconds * RATE / HOP)


def available_units(reference_samples, predicted_speed):
    available_seconds = max(1.0, 22.0 - reference_samples / RATE)
    return max(1, int(predicted_speed * available_seconds))


def crossfade(waves, seconds):
    final = waves[0]
    for nxt in waves[1:]:
        count = min(int(seconds * RATE), len(final), len(nxt))
        if count <= 0:
            final = np.concatenate((final, nxt))
            continue
        out = np.linspace(1.0, 0.0, count)
        inc = np.linspace(0.0, 1.0, count)
        overlap = final[-count:] * out + nxt[:count] * inc
        final = np.concatenate((final[:-count], overlap, nxt[count:]))
    return final


def generate():
    duration_rows = []
    sample_rates = [100, 256, 8192, 24000, 25000, 128000, 350000, 485678, 528000]
    rates = [0.5, 0.8, 1.0, 1.3, 1.7, 2.0, 3.25]
    for i in range(2200):
        reference_samples = RNG.choice(sample_rates) if i % 3 == 0 else RNG.randint(257, 528000)
        units = RNG.randint(1, 70)
        speed = RNG.randint(1, 32) * 0.25
        user_speed = RNG.choice(rates)
        limit = available_units(reference_samples, speed)
        frames = predicted_frames(reference_samples, units, speed, user_speed)
        duration_rows.append(f'{reference_samples}\t{units}\t{speed}\t{user_speed}\t{limit}\t{frames}\n')
    (ROOT / 'cross_lingual_duration_python_expected.tsv').write_text(''.join(duration_rows), encoding='utf-8')

    rows = []
    for i in range(128):
        n = RNG.randint(1, 4)
        waves = []
        for _ in range(n):
            values = np.array([RNG.uniform(-0.9, 0.9) for _ in range(RNG.randint(1, 20))], dtype=np.float32)
            waves.append(values)
        seconds = RNG.choice([0.0, 1 / RATE, 3 / RATE, 7 / RATE, 0.0003, 0.0004])
        expected = crossfade(waves, seconds)
        flat_waves = ';'.join(','.join(f'{float(x):.9g}' for x in wave) for wave in waves)
        rows.append(f'{seconds:.17g}\t{flat_waves}\t' + ','.join(f'{float(x):.12g}' for x in expected) + '\n')
    (ROOT / 'cross_lingual_crossfade_numpy_expected.tsv').write_text(''.join(rows), encoding='utf-8')
    print(f'Cross-Lingual: {len(duration_rows)} duration and {len(rows)} crossfade cases')


if __name__ == '__main__':
    generate()
