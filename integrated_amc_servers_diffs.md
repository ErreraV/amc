**Integrated AMC Servers — Key Differences**

**Files compared:** [src/amc/integrated_amc_gan_rl.py](src/amc/integrated_amc_gan_rl.py), [src/amc/integrated_amc_gan_test.py](src/amc/integrated_amc_gan_test.py), [src/amc/integrated_amc_gan_enzo.py](src/amc/integrated_amc_gan_enzo.py)

**Flow state management:**
- **`rl` / `enzo`**: use `preselected_modulations` (store modulation + predicted_snr).
- **`test`**: uses `predicted_snrs` (stores only predicted_snr timestamped).

**Prediction storage & usage:**
- **`rl` / `enzo`**: proactively compute a modulation for t+1 and cache the modulation with predicted SNR; when valid they may reuse the cached modulation directly.
- **`test`**: caches only the numeric predicted SNR and later uses that SNR as the RL agent input (DRL decision uses predicted SNR rather than a preselected modulation).

**Preprocessor / denormalization:**
- **`rl`**: implements lightweight manual denormalization (uses stored mean/scale floats) and fast numpy arithmetic.
- **`test` / `enzo`**: rely on the scaler objects (`scaler_snr` / `scaler_aux`) and call their `transform`/`inverse_transform` methods; `enzo` includes more robust denorm with logging and a fallback scaling.

**Proactive prediction behavior:**
- **`enzo`**: requests confidence interval (`return_confidence=True`) when available and logs history/aux and conf; clamps/validates predictions more conservatively (different clamp ranges in places).
- **`rl`**: stores predicted SNR but originally also had logic to run RL to preselect modulation (some code paths differ between versions).
- **`test`**: focuses on predicted-snr-first architecture (DRL uses predicted SNR directly).

**Selection & RL callsites:**
- Function naming differs: `_select_modulation` vs `_select_modulation_current_snr` vs `_select_modulation` variants; some files call RL on `decision_snr` (predicted or current), others call RL on current SNR or reuse cached modulation.

**Threading / inference lock:**
- `rl` and `test` contain an `inference_lock` to serialize GPU/RL/gan access; `enzo` moves or omits locking in some callsites and relies more on serialized single-threaded or explicit logging behavior.

**Logging & instrumentation:**
- `enzo` and `test` add more `logger.info/debug` calls (prediction diagnostics, fallback warnings, RL updates). `rl` is lighter-weight in log verbosity.

**Sionna / fallback differences:**
- Error handling and debug logging around Sionna availability differ (e.g., `enzo` logs Sionna fallback debug message). The returned fields are mostly consistent, but throughput calculations differ (some versions multiply by `50.0` while others do not).

**Model save / load:**
- All try to load enhanced GAN and RL agent; `enzo` uses slightly different save semantics and removes the `inference_lock` around RL model save in some variants.

**Server stats / bookkeeping keys:**
- `cached_preselections` / `preselected_modulations` vs `predicted_sns` naming differs; prediction-accuracy counting and when `predictions_accurate` is incremented varies.

**Small API/behavioral differences:**
- Decision `method` strings differ (`proactive_rl_with_gan`, `preselected_from_gan_prediction`, `drl_agent_with_gan_prediction`, `rl_agent_current_snr`, etc.).
- Minor changes in protection/clamping thresholds and exception fallbacks.

**Recommendation (if you want to harmonize):**
- Pick intended architecture: (A) cache modulation+SNR (use preselected_modulations), or (B) cache numeric predicted SNR and always run RL with that SNR. Then standardize names (`store_preselected_modulation` vs `store_predicted_snr`), adjust `get_server_stats()` keys, and align denorm logic to either use scalers or manual mean/scale consistently.

**End of summary**
