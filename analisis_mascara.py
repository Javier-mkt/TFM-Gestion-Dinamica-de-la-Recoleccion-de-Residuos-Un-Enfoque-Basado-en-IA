import numpy as np

def _is_vec_env(env) -> bool:
    return hasattr(env, "num_envs") or hasattr(env, "envs") or hasattr(env, "venv")

def _obs_to_batch(obs_dict):
    """Convierte obs (dict) a formato batcheado (B, ...) aunque B=1."""
    out = {}
    for k, v in obs_dict.items():
        arr = np.asarray(v)
        if k in ("mascara_tipo", "mascara_destino") and arr.ndim == 1:
            arr = arr[None, ...]              # -> (1, 2) o (1, N)
        elif k == "mask2_table" and arr.ndim == 2:
            arr = arr[None, ...]              # -> (1, 2, N)
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)
        out[k] = arr
    return out

def _count_valid_actions(mask_tipo, mask2_table):
    """
    mask_tipo:    (2,) bool/int
    mask2_table:  (2, N) bool/int
    Devuelve:
      - valid_types: int
      - valid_dests_per_type: list len=2
      - total_joint: int  (sum_t valid_dests[t] solo para t con tipo válido)
    """
    mt = (mask_tipo.astype(bool))
    md = (mask2_table.astype(bool))
    valid_types = int(mt.sum())
    valid_dests_per_type = [int(md[0].sum()), int(md[1].sum())]
    total_joint = int((md[0].sum() if mt[0] else 0) + (md[1].sum() if mt[1] else 0))
    return valid_types, valid_dests_per_type, total_joint

def _sample_legal_action(mask_tipo, mask2_table, rng):
    """Devuelve (tipo, destino) aleatorio válido. Si no hay ninguno, devuelve (0,0) y marca flag False."""
    mt = np.where(mask_tipo.astype(bool))[0]
    if mt.size == 0:
        return (0, 0), False
    t = rng.choice(mt)
    dests = np.where(mask2_table[t].astype(bool))[0]
    if dests.size == 0:
        return (t, 0), False
    d = rng.choice(dests)
    return (int(t), int(d)), True

def analizar_mascaras(env, pasos=3000, seed=123):
    """
    Recorre el entorno ejecutando acciones legales aleatorias y reporta:
      - % de estados con acciones conjuntas válidas = 0 (bug/edge-case)
      - % de estados deterministas (solo 1 acción válida conjunta)
      - medias y percentiles del nº total de acciones válidas conjuntas
      - distribución de 'nº de tipos válidos' y 'nº de destinos válidos por tipo'
    """
    rng = np.random.default_rng(seed)
    is_vec = _is_vec_env(env)

    # RESET
    if is_vec:
        obs = env.reset()
    else:
        obs, _ = env.reset()

    # Asegura formato batcheado
    obs_b = _obs_to_batch(obs)

    stats_total = 0
    zero_joint = 0
    deterministic_joint = 0
    joint_card_list = []
    valid_types_list = []
    valid_dests_t0, valid_dests_t1 = [], []

    while stats_total < pasos:
        mt = obs_b["mascara_tipo"]           # (B,2)
        m2 = obs_b["mask2_table"]            # (B,2,N)
        B = mt.shape[0]

        # Contabiliza por elemento del batch
        for b in range(B):
            vt, vd_per_t, tj = _count_valid_actions(mt[b], m2[b])
            stats_total += 1
            valid_types_list.append(vt)
            valid_dests_t0.append(vd_per_t[0])
            valid_dests_t1.append(vd_per_t[1])
            joint_card_list.append(tj)
            if tj == 0:
                zero_joint += 1
            elif tj == 1:
                deterministic_joint += 1

            if stats_total >= pasos:
                break

        # Construye acciones legales aleatorias para avanzar
        acciones = []
        for b in range(B):
            (t, d), ok = _sample_legal_action(mt[b], m2[b], rng)
            acciones.append([t, d])
        acciones = np.asarray(acciones, dtype=np.int64)

        # STEP
        if is_vec:
            obs, _, dones, _ = env.step(acciones)
            # si hay algún done, reset implícito del vecenv; seguimos con obs
        else:
            obs, _, term, trunc, _ = env.step(acciones[0])
            if term or trunc:
                obs, _ = env.reset()

        obs_b = _obs_to_batch(obs)

    # ---- Reporte ----
    joint_arr = np.array(joint_card_list, dtype=np.int32)
    vt_arr = np.array(valid_types_list, dtype=np.int32)
    v0_arr = np.array(valid_dests_t0, dtype=np.int32)
    v1_arr = np.array(valid_dests_t1, dtype=np.int32)

    def _pct(x): return 100.0 * x / max(1, stats_total)

    resumen = {
        "estados_total": int(stats_total),
        "estados_con_0_acciones": int(zero_joint),
        "pct_con_0_acciones": _pct(zero_joint),
        "estados_deterministas_(1_accion)": int(deterministic_joint),
        "pct_deterministas": _pct(deterministic_joint),
        "joint_acciones_validas_media": float(joint_arr.mean()) if stats_total else 0.0,
        "joint_acciones_validas_p50": float(np.percentile(joint_arr, 50)) if stats_total else 0.0,
        "joint_acciones_validas_p90": float(np.percentile(joint_arr, 90)) if stats_total else 0.0,
        "joint_acciones_validas_p99": float(np.percentile(joint_arr, 99)) if stats_total else 0.0,
        "tipos_validos_media": float(vt_arr.mean()) if stats_total else 0.0,
        "destinos_validos_tipo0_media": float(v0_arr.mean()) if stats_total else 0.0,
        "destinos_validos_tipo1_media": float(v1_arr.mean()) if stats_total else 0.0,
    }

    print("\n=== Diagnóstico de máscaras ===")
    for k, v in resumen.items():
        if "pct" in k:
            print(f"{k:34s}: {v:6.2f}%")
        else:
            print(f"{k:34s}: {v}")

    # Extra: histogramas resumidos (texto) de cardinalidades conjuntas
    if stats_total:
        vals, counts = np.unique(joint_arr, return_counts=True)
        print("\nHistograma de nº de acciones conjuntas válidas (top 10):")
        top = np.argsort(-counts)[:10]
        for i in top:
            print(f"  {int(vals[i]):>4d}  -> {int(counts[i])} estados ({_pct(counts[i]):.2f}%)")

    return resumen
