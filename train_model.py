#!/usr/bin/env python3
"""
train_model.py

Funzioni di allenamento per il modello policy+value usato nel progetto scacchi.

 Usage example:
    python train_model.py --model chess_elo_model_V0 --dataset all_positions_jul2014_npz/positions_jul2014_game1_game1500.npz --save_to model_versions/chess_elo_model_V1 --epochs 20
"""

import argparse
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import json

def load_npz_dataset(npz_path):
    print("Carico dataset:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X_boards = data["X_boards"]   # (N,8,8,13)
    X_eloside = data["X_eloside"] # (N,5)
    y = data["y"]                 # (N,)
    y_value = data["y_value"]     # (N,)
    legal_indices = data.get("legal_indices", None) # (N,num_classes)

    print("Shapes:", X_boards.shape, X_eloside.shape, y.shape, y_value.shape, legal_indices.shape)
    print("Type:", X_boards.dtype, X_eloside.dtype, y.dtype, y_value.dtype, legal_indices.dtype)

    num_classes = len(legal_indices[0])
    print("Numero classi (mosse) =", num_classes)
    return X_boards, X_eloside, y, y_value, num_classes, legal_indices 

def compute_bests(history_dict):
    bests = {}
    for k, vals in history_dict.items():
        try:
            arr = np.array(vals, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        kl = k.lower()
        # regola semplice: per metriche "loss"/"mse"/"mae" scegli il minimo, altrimenti il massimo
        if ("loss" in kl) or ("mse" in kl) or ("mae" in kl):
            idx = int(np.argmin(arr))
            bests[k] = {"best_value": float(arr[idx]), "epoch": int(idx+1), "mode": "min"}
        else:
            idx = int(np.argmax(arr))
            bests[k] = {"best_value": float(arr[idx]), "epoch": int(idx+1), "mode": "max"}
    return bests

def make_tf_dataset(X_boards, X_eloside, y, y_value, X_legal_indices, batch_size=32, shuffle=True, alpha=1.0):
    N = X_boards.shape[0]
    y_policy = y.astype(np.int32)
    y_val = y_value.astype(np.float32)

    policy_weights = 1.0 + alpha * y_val
    policy_weights = policy_weights / float(np.mean(policy_weights))
    value_weights = np.ones_like(y_val, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(((X_boards, X_eloside, X_legal_indices), (y_policy, y_val), (policy_weights, value_weights)))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, N), reshuffle_each_iteration=True)

    def map_fn(inputs, targets, weights):
        board, elos, indi = inputs
        policy_label, value_label = targets
        policy_w, value_w = weights
        board = tf.cast(board, tf.float32)
        elos  = tf.cast(elos, tf.float32)
        indi  = tf.cast(indi, tf.float32)
        x = {"board": board, "eloside": elos, "legal_indices": indi}
        y = {"policy": policy_label, "value": value_label}
        sw = (policy_w, value_w)
        return x, y, sw

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model (es. chess_elo_model without .keras)")
    parser.add_argument("--dataset", type=str, required=True, help="dataset .npz (es. dataset_from_fen.npz)")
    parser.add_argument("--save_to", type=str, required=True, help="Path to save updated model without .keras")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_per", type=float, default=0.9, help="Train/val percentage")
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for policy_weights / value_weights")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--policy_weight", type=float, default=1.0)
    parser.add_argument("--value_weight", type=float, default=0.5)
    parser.add_argument("--monitor", default="val_policy_loss")
    parser.add_argument("--compile", default=True)
    parser.add_argument("--validation", default=None, help="validation dataset")
    parser.add_argument("--validation_indices")
    args = parser.parse_args()


    if args.validation:
        csv_path = (args.dataset).replace("npz", "csv")
        with open("validation_selected_indices.json", "r", encoding="utf-8") as f:
            indices = json.load(f)
        this_ind = indices[csv_path]

    X_boards, X_eloside, y, y_value, _, legal_indices = load_npz_dataset(args.dataset)
    N = X_boards.shape[0]

    # Load model
    print(f"Loading model from {args.model}.keras ...")
    
    model = tf.keras.models.load_model(args.model + ".keras", compile=args.compile)

    if args.compile == False:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss={"policy": tf.keras.losses.SparseCategoricalCrossentropy(),"value": tf.keras.losses.MeanSquaredError()},
                  loss_weights={"policy": args.policy_weight, "value": args.value_weight},
                  metrics={"policy": tf.keras.metrics.SparseCategoricalAccuracy(name="policy_acc"),"value": tf.keras.metrics.MeanSquaredError(name="value_mse")})

    if args.validation:        
        X_boards_val, X_eloside_val, y_val, y_value_val, _, legal_indices_val = load_npz_dataset(args.validation)

        Xb_train, Xe_train, y_train, yv_train, Li_train = [], [], [], [], []
        for i in range(N):
            if i not in this_ind:
                Xb_train.append(X_boards[i])
                Xe_train.append(X_eloside[i])
                y_train.append(y[i])
                yv_train.append(y_value[i])
                Li_train.append(legal_indices[i])
        Xb_train = np.array(Xb_train, dtype=np.uint8)
        Xe_train = np.array(Xe_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        yv_train = np.array(yv_train, dtype=np.float32)
        Li_train = np.array(Li_train, dtype=np.uint8)

        train_ds = make_tf_dataset(Xb_train, Xe_train, y_train, yv_train, Li_train, batch_size=args.batch_size, shuffle=True, alpha=args.alpha)
        val_ds = make_tf_dataset(X_boards_val, X_eloside_val, y_val, y_value_val, legal_indices_val, batch_size=args.batch_size, shuffle=True, alpha=args.alpha)
    else:
        idx = np.arange(N)
        np.random.shuffle(idx)
        split = int(args.train_per * N)
        train_idx, val_idx = idx[:split], idx[split:]

        Xb_train, Xe_train, y_train, yv_train, Li_train = X_boards[train_idx], X_eloside[train_idx], y[train_idx], y_value[train_idx], legal_indices[train_idx]
        Xb_val, Xe_val, y_val, yv_val, Li_val = X_boards[val_idx], X_eloside[val_idx], y[val_idx], y_value[val_idx], legal_indices[val_idx]

        train_ds = make_tf_dataset(Xb_train, Xe_train, y_train, yv_train, Li_train, batch_size=args.batch_size, shuffle=True, alpha=args.alpha)
        val_ds = make_tf_dataset(Xb_val, Xe_val, y_val, yv_val, Li_val, batch_size=args.batch_size, shuffle=False, alpha=args.alpha)

    cb = []
    checkpoint_path = args.model + "_best.keras"
    cb.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False))
    cb.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=args.monitor, factor=0.5, patience=2, min_lr=1e-6, mode="min"))
    cb.append(tf.keras.callbacks.EarlyStopping(monitor=args.monitor, patience=6, restore_best_weights=True, mode="min"))

    print("Inizio training: epochs", args.epochs, "batch_size", args.batch_size)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb, verbose=1)

    model.save(args.save_to + ".keras")
    print("Modello salvato in:", args.save_to + ".keras")

    # prepara la entry di history da appendere (includo timestamp e alcuni argomenti utili)
    entry = {
        "timestamp_utc": datetime.now().isoformat() + "Z",
        # salvo gli args (tutti tranne save_to per evitare ridondanza)
        "args": {k: v for k, v in vars(args).items() if k not in ("save_to",)},
        "history": history.history,
        "best": compute_bests(history.history)
    }

    history_path = args.save_to + "_history.npy"

    # se esiste, caricalo e appendi; altrimenti crea lista nuova
    if os.path.exists(history_path):
        try:
            existing = np.load(history_path, allow_pickle=True)
            # converti il contenuto caricato in una lista python
            if isinstance(existing, np.ndarray):
                existing_py = existing.tolist()
            else:
                existing_py = existing

            # se era un singolo dict (vecchio formato), fallo diventare lista
            if isinstance(existing_py, dict):
                master_list = [existing_py]
            elif isinstance(existing_py, list):
                master_list = existing_py
            else:
                # fallback: mettilo come singolo elemento in lista
                master_list = [existing_py]
        except Exception as e:
            # in caso di problemi di lettura, crea nuova lista ma segnala l'errore
            print("Warning: impossibile leggere history esistente:", e)
            master_list = []
    else:
        master_list = []

    # aggiungi la entry corrente e risalva
    master_list.append(entry)
    np.save(history_path, master_list, allow_pickle=True)
    print(f"History aggiornata e salvata in: {history_path}  (entries totale = {len(master_list)})")


if __name__ == "__main__":
    main()
