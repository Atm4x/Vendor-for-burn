# Stage 51 — CUDA Graph и планирование NSF на уровне CubeCL

**Дата:** 11 ноября 2026. **Репозиторий:** `Atm4x/Vendor-for-burn`, экспериментальный PR #2. **Статус:** код A/B-профайлера подготовлен, `cargo check` CPU/CUDA/WGPU пройден; реальная производительность CUDA Graph на RTX 4060 ожидает измерения. Production NSF не изменён.

## 1. Почему `CUBECL_CPU_MAX_TASKS=32` нельзя буквально перенести в CUDA

Изучены реальные реализации `cubecl-runtime`, `cubecl-server`, `cubecl-cpu`, `cubecl-cuda`:

- CPU использует `SchedulerMultiStream` с лимитом `max_tasks` (стандартное значение 8), после чего исполняет очередной batch. Для CPU этот путь может блокировать поток до завершения зависимых задач. Stage 44 показал, что изменение размера группы и отключение вложенного Rayon-параллелизма улучшали latency синтетического NSF на виртуализированном CPU.
- CUDA использует `CudaServer` и `MultiStream<CudaStreamBackend>`, CUDA events и асинхронную отправку kernel launches. На этом пути **нет того же блокирующего CPU `max_tasks=8`**.
- CUDA `flush()` в текущем сервере выполняет обслуживание отложенных освобождений и памяти, но не является эквивалентом ожидания и запуска CPU batch. Stage 42/45 подтвердил, что принудительные `flush()` не помогают GPU в исследованном графе.

**Вывод:** прямой перенос числа 32 в CUDA бессмысленен. Возможные CUDA-native аналоги для сокращения стоимости множества host-side запусков: пакетирование при dispatch либо CUDA Graph Capture/Replay. Второй механизм уже реализован в CubeCL и обеспечивает воспроизведение нескольких kernel launches одним графовым dispatch, без изменения арифметики Conv1d.

## 2. Новый эксперимент Stage 51

Новый файл: `models/tts-with-rvc-rs/src/bin/profile_nsf_graph_stage51.rs`.

Для каждой длины NSF (по умолчанию 64, 256 и 1024 кадра):

1. Декодер и веса загружаются один раз, вход, harmonic source и conditioning создаются **один раз** и удерживаются живыми в рамках эксперимента.
2. Выполняется контрольный обычный forward.
3. Выполняется `burn::tensor::capture(&device, || decoder.forward_with_execution(...))`, включая подготовительные прогоны и попытку записи hardware graph. Стоимость capture измеряется **отдельно**.
4. Проверяется `graph.is_hardware()`. Если `false`, backend повторяет closure программно: **это не аппаратный CUDA Graph**; такое отношение задержек запрещено трактовать как GPU-ускорение.
5. `ordinary forward` и `unsafe { graph.replay() }` чередуются в одном процессе. На каждом раунде полное FP32-выходное значение читается на CPU, чтобы дождаться завершения GPU.
6. Сохраняются времена dispatch, readback и полной операции; median, сырые примеры, точное сравнение FP32 и максимальная ошибка. Отдельный минимальный граф `(x + 1)` проверяет, поддерживается ли аппаратный capture вообще, независимо от сложности NSF.

Проверяются конфигурации:

- `baseline`: обычная Conv1d, Sequential residual execution;
- `prepacked`: оптимизация Stage 49 с единоразовой подготовкой весов;
- `packed`: специализированная CubeCL temporal-pair Conv1d;
- `quad_interior`: специализированная temporal-quad с оптимизацией внутренних блоков;
- `quad`: дополнительный опциональный вариант.

PowerShell runner: `research/stage37-windows-gpu-bench/windows/Run-NSF-Graph.ps1`. Два независимых набора процессов с обратным порядком вариантов; внутри каждого процесса обычные forward и replay чередуются. Диагностика и JSON записываются в `NSF_CUDA_GRAPH_STAGE51_RESULTS.zip`.

## 3. Проверки, проведённые до публикации

| Проверка | Статус |
|---|---|
| Rust 1.98.1, `cargo check --offline --locked`, CPU | PASS |
| Rust 1.98.1, `cargo check --offline --locked`, CUDA | PASS |
| Rust 1.98.1, `cargo check --offline --locked`, WGPU | PASS |
| Linux CPU JIT `release`, полный NSF (4/64 кадров) | PASS |
| CPU-выходы normal vs replay | Побитовое совпадение (`max_abs_error = 0`) |
| CPU `graph.is_hardware()` | `false`, ожидаемый software fallback |
| Windows CUDA EXE | Предназначен для GitHub Actions; отдельный hardware runtime test ещё не выполнен |
| CUDA Graph hardware capture + RTX 4060 A/B | **Не измерено** |

CPU JIT smoke: на 64 кадрах normal ≈16.07 мс и fallback replay ≈16.07 мс. На 4 кадрах большой шум между раундами (примерно 16–36 мс); эти значения не используются как прогноз CUDA Graph.

## 4. Ключевые ограничения безопасности и корректности

- **Статические входы:** граф захватывает адреса исходных GPU-буферов. Benchmark переигрывает тот же вход, а не валидирует постоянно меняющееся аудио. Это чистая проверка overhead для фиксированной структуры NSF.
- **Время capture:** зависит от прогрева и потребления памяти; отдельно записывается в отчёт. Прирост при многократном replay окупает эту стоимость только при достаточном количестве повторений.
- **Безопасность raw pointers:** `Graph::replay()` — `unsafe`, потому что граф использует буферы по адресам, записанным при capture. В тесте входные tensors, decoder/weights и сам graph живут до окончания каждого replay, а output readback синхронизирует выполнение. Нельзя удалять входы или изменять их из другого stream.
- **Production-допуск:** для меняющегося аудио нужно гарантированное in-place обновление **того же GPU-буфера**, правильная same-stream ordering и повторное сравнение выходов на нескольких различных входах. `Tensor::inplace()` в Burn **не обещает** сохранение того же allocation; поэтому простое вызвание `inplace` не является доказательством корректного обновления graph inputs.
- **Fallback:** `is_hardware=false` или ошибка capture означает, что результат по CUDA Graph не получен, даже если closure продолжает работать.
- **Это не kernel fusion:** CUDA Graph сокращает CPU-side launch overhead; оно не меняет вычислительную сложность Conv1d, число арифметических операций или локальность памяти.

## 5. Запуск на RTX 4060

После успешного Windows GitHub Actions workflow `RVC NSF Stage51 CUDA Graph A/B package` скачать artifact `rvc-nsf-cuda-graph-stage51-windows-x64`, распаковать и запустить:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-NSF-Graph.ps1 -Iterations 20 -Warmup 5 -Frames "64,256,1024"
```

Проверка дополнительного ядра:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-NSF-Graph.ps1 -Iterations 20 -Warmup 5 -Frames "64,256,1024" -Variants "prepacked,packed,quad,quad_interior"
```

**Передать:** `NSF_CUDA_GRAPH_STAGE51_RESULTS.zip`. В результатах смотреть прежде всего `basic_graph_hardware`, `cases[].hardware_graph`, `ordinary_dispatch_median_ms` vs `replay_dispatch_median_ms`, `ordinary_median_ms` vs `replay_median_ms`, `capture_setup_ms` и FP32 parity. Если обычный маленький граф аппаратный, а большой NSF — нет, причина вероятнее в захвате/аллокациях, разных streams либо жизненном цикле временных буферов NSF; это отдельное направление Stage 52.

## 6. План следующего R&D

1. Отделить GPU kernel execution от host dispatch через `Device::profile`/CUDA events; сравнить обычный forward и hardware replay на одном stream.
2. Если hardware capture не работает на NSF, сузить problematic operation методом инкрементального добавления стадий, **не переключая граф на программный fallback молча**.
3. Построить безопасную схему обновления persistent CUDA входов для разных кадров с фиксированной формой, подтвердить полную FP32 parity на каждом изменении.
4. Измерить окупаемость capture в числе повторных forward, VRAM overhead и рост memory pool.
5. В отдельном опыте варьировать CUDA stream policy (`single` против `per-thread`) и ограничение `max_streams` на одинаковом NSF-графе. Это архитектурно ближе к анализу многопоточной конструкции Stage 44, чем перенос CPU batch-count.
6. Только после этих проверок рекомендовать CUDA Graph для production — поверх доказанного Stage49 Prepacked и отдельно от вариантов Packed/QuadInterior.

## 7. Дополнительный cross-backend эксперимент: native stream policy

Исходный CubeCL предоставляет переносимый `StreamPolicy` и настройки `[streaming]` в `cubecl.toml`, подхватываемые при первом создании runtime:

- `policy = "per-thread", max_streams = 128`: стандартный CUDA путь с отдельными логическими потоками отправки для разных CPU threads;
- `policy = "single", max_streams = 1`: единый CUDA stream для операций всех CPU threads.

Это **не batch=32**: настройка определяет распределение команд и стоимость межпоточных GPU-зависимостей, а не число задач в CPU scheduler. Для проверки CUDA добавлен второй runner `Run-NSF-Streams.ps1` (в том же Windows artifact). Он использует существующий Stage49 Prepacked NSF, сравнивает `Parallel` и `Sequential` с обеими stream-политиками, пересоздаёт устройство в отдельном процессе на каждую комбинацию, проверяет полные FP32-выходы и разворачивает порядок комбинаций во втором проходе. Каждому процессу выдаётся свой каталог с отдельным `cubecl.toml`.

**Локальный предварительный CPU-тест** с `CUBECL_CPU_MAX_TASKS=32`, 5 CPU, 64 кадра, 10 измерений после 4 прогревов, три независимых процесса для каждой комбинации:

| Stream policy | Residuals | Времена трёх запусков, мс | Медиана, мс |
|---|---|---|---:|
| `per-thread` | Parallel | 30,06 / 37,57 / 38,88 | 37,57 |
| `single` | Parallel | 15,98 / 37,09 / 16,34 | 16,34 |
| `per-thread` | Sequential | 16,01 / 16,26 / 16,15 | 16,15 |
| `single` | Sequential | 16,04 / 16,25 / 16,01 | 16,04 |

Число выходных элементов и контрольные суммы для всех 12 прогонов совпали, повторная численная ошибка внутри каждого прогона — ноль. Это **не** полная побитовая cross-process проверка, поскольку данный smoke не экспортировал FP32-файлы. Результаты сильно варьируют (один `single + Parallel` запуск занял 37 мс), поэтому нельзя заявить устойчивое 2,3× ускорение только от stream policy; это обоснование для A/B на GPU.

Windows запуск второго теста:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-NSF-Streams.ps1 -Iterations 20 -Warmup 5 -Frames "64,256,1024"
```

Архив: `NSF_CUDA_STREAMS_STAGE51_RESULTS.zip`. Он отвечает на вопрос о влиянии **CUDA-native** распределения stream-зависимостей, тогда как `Run-NSF-Graph.ps1` измеряет сокращение CPU-side запусков через CUDA Graph. Оба опыта сохраняют неизменный математический NSF.
