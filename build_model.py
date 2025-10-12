#!/usr/bin/env python3
"""
build_model.py

Funzioni per costruire, salvare il modello policy+value usato nel progetto scacchi.

Funzioni principali:
 - build_model(...)    -> costruisce e ritorna il modello non compilato (pesi casuali)
 - compile_model(...)  -> compila il modello con optimizer, loss e metriche

 Usage example:
    python build_model.py all_positions_jul2014_npz/positions_jul2014_game1_game1500.npz --model_out chess_elo_model_V0
"""

import tensorflow as tf
import numpy as np
import argparse

def load_npz_dataset_shape(npz_path):
    print("Carico dataset:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X_boards_shape = data["X_boards"].shape                          # (N,8,8,13)
    X_eloside_shape = data["X_eloside"].shape                        # (N,5)
    y_shape = data["y"].shape                                        # (N,)
    y_value_shape = data["y_value"].shape                            # (N,)
    legal_indices_shape = data.get("legal_indices", None).shape      # (N,num_classes)

    print("Shapes:", X_boards_shape, X_eloside_shape, y_shape, y_value_shape, legal_indices_shape)

    num_classes = int(legal_indices_shape[1])
    print("Numero classi (mosse) =", num_classes)
    return X_boards_shape, X_eloside_shape, y_shape, y_value_shape, num_classes, legal_indices_shape

def build_model(input_shape=(8,8,13), eloshape=(5,), legal_shape=(1,),  num_classes=1000):
    """
    Costruisce e ritorna un modello Keras multi-output:
      - input 'board'  : tensor (H,W,C) es. (8,8,13)
      - input 'eloside': tensor (5,)  -> [white_elo_norm, black_elo_norm, side_to_move]
      - input 'legal_moves_in': tensor (num_classes,)
      - output 'policy': softmax su num_classes (probabilità sulle mosse)
      - output 'value' : sigmoid scalare [0,1] (prob vittoria per side_to_move)

    Nota: la funzione *non compila* il modello (nessun optimizer/loss impostato)
    """
    # input layers
    board_in = tf.keras.Input(shape=input_shape, name="board")
    elos_in  = tf.keras.Input(shape=eloshape, name="eloside")
    legal_moves_in = tf.keras.Input(shape=legal_shape, name="legal_indices")

    # --- feature extractor per la board (CNN)
    x = board_in
    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)   # (8,8) -> (4,4)

    x = tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)   # (4,4) -> (2,2)

    x = tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # -> (256,)

    # --- elaborazione ELO/side (MLP leggero)
    e = tf.keras.layers.Dense(64, activation="relu")(elos_in)
    e = tf.keras.layers.BatchNormalization()(e)

    # --- elaborazione ELO/side (MLP leggero)
    i = tf.keras.layers.Dense(64, activation="relu")(legal_moves_in)
    i = tf.keras.layers.BatchNormalization()(i)

    # --- concateniamo le feature e le passiamo a un paio di dense
    c = tf.keras.layers.Concatenate()([x, e, i])
    c = tf.keras.layers.Dense(512, activation="relu")(c)
    c = tf.keras.layers.Dropout(0.4)(c)
    c = tf.keras.layers.Dense(256, activation="relu")(c)

    # --- teste finali
    policy_logits = tf.keras.layers.Dense(num_classes, activation=None, name="policy_logits")(c)
    value = tf.keras.layers.Dense(1, activation="sigmoid", name="value")(c)

    # Apply mask to logits BEFORE softmax:
    # legal_moves_in is (None, num_classes) with 0/1 floats
    # masked_logits = policy_logits + (1.0 - legal_moves_in) * (-1e9)
    neg_inf = tf.constant(-1e9, dtype=tf.float32)
    masked_logits = policy_logits + (1.0 - legal_moves_in) * neg_inf

    # final policy probabilities (softmax over masked logits)
    policy = tf.keras.layers.Activation("softmax", name="policy")(masked_logits)

    model = tf.keras.Model(inputs=[board_in, elos_in, legal_moves_in], outputs=[policy, value], name="policy_value_net_masked")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="dataset .npz (es. dataset_from_fen.npz)")
    parser.add_argument("--model_out", required=True, default="chess_elo_model")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--policy_weight", type=float, default=1.0)
    parser.add_argument("--value_weight", type=float, default=0.5)
    args = parser.parse_args()

    X_boards_shape, X_eloside_shape, _, _, num_classes, _ = load_npz_dataset_shape(args.npz)

    model = build_model(input_shape=X_boards_shape[1:], eloshape=(X_eloside_shape[1],), legal_shape=(num_classes,), num_classes=num_classes)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss={"policy": tf.keras.losses.SparseCategoricalCrossentropy(),"value": tf.keras.losses.MeanSquaredError()}, 
                  loss_weights={"policy": args.policy_weight, "value": args.value_weight},
                  metrics={"policy": tf.keras.metrics.SparseCategoricalAccuracy(name="policy_acc"),"value": tf.keras.metrics.MeanSquaredError(name="value_mse")})
    
    model.save(args.model_out + ".keras")
    print("Modello salvato in:", args.model_out + ".keras")