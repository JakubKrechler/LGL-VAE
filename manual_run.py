"""Manual, step-by-step runner for learning the LGL-VAE project.

This file is intended as a study scaffold. It keeps the code split into
independent pieces so you can run and modify each part separately.

Examples:
    python manual_run.py --step 1
    python manual_run.py --step 2 3 4 5
    python manual_run.py --step 2,3,4,5
    python manual_run.py --step 8 --dca-model-out trained_dca_manual.pkl
    python manual_run.py --step 9 --model-out trained_vae_manual
    python manual_run.py --step 12 --dca-pkl trained_vae_manual/latent_generative_landscape.pkl
    python manual_run.py --step 9 --model-out trained_vae_manual --dca-model-out trained_dca_manual.pkl
    python manual_run.py --step 13 --model-out trained_vae_manual --dca-pkl trained_vae_manual/latent_generative_landscape.pkl
    python manual_run.py --step 14 --model-out trained_vae_manual --sequence 0 --similarity-threshold 80
    python manual_run.py --step 14 --model-out trained_vae_manual --sequence 0 --similarity-threshold 80 --three-dimensional
    python manual_run.py --step 15 --model-out trained_vae_manual --dca-pkl trained_vae_manual/latent_generative_landscape.pkl
    python manual_run.py --step all

Direct Python use:
    from manual_run import run_steps
    run_steps([2, 3, 4, 5], fasta_path="globin_demo/training_sequences.fasta")
"""

from __future__ import annotations

import argparse
import bz2
import base64
import csv
import io
import math
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Bio import SeqIO

from model.generator import get_fasta_file_dimensions, read_fasta_as_one_hot_encoded, seq_code
from model.model import VAE

DEFAULT_FASTA = "globin_demo/training_sequences.fasta"


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_steps(raw_steps: list[str]) -> list[int]:
    """Accepts values like ["1", "2,3", "4", "all"] and returns ints."""
    flat: list[int] = []
    for item in raw_steps:
        value = str(item).strip().lower()
        if value == "all":
            return list(range(1, 16))
        for part in value.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            flat.append(int(cleaned))
    return flat


def step_2_inspect_fasta(path: str) -> None:
    """Step 2: basic FASTA info."""
    print_header("Krok 2: zkontroluj, kolik sekvencí a jak dlouhá je každá")
    num_sequences, seq_len = get_fasta_file_dimensions(path)
    print(f"Cesta: {path}")
    print(f"Počet sekvencí: {num_sequences}")
    print(f"Délka sekvence: {seq_len}")

    for i, record in enumerate(SeqIO.parse(path, "fasta")):
        if i >= 3:
            break
        print(f"Příklad sekvence {i}: header={record.id}, délka={len(record.seq)}")


def step_3_preview_sequences(path: str, limit: int = 2) -> None:
    """Step 2: look at raw FASTA records."""
    print_header("Krok 3: prohlédni si RAW data v FASTA formátu")
    for i, record in enumerate(SeqIO.parse(path, "fasta")):
        if i >= limit:
            break
        print(f"[{i}] header: {record.id}")
        print(f"    sekvence: {str(record.seq)[:80]}")


def step_4_encode_one_sequence(path: str, index: int = 0) -> None:
    """Step 3: convert one sequence to one-hot representation."""
    print_header("Krok 4: jedna sekvence -> one-hot matice")
    for i, encoded in enumerate(read_fasta_as_one_hot_encoded(path)):
        if i != index:
            continue
        print(f"Vybraná sekvence index: {i}")
        print(f"Shape po one-hot: {encoded.shape}")
        print(f"První 10 hodnot: {encoded[:10]}")
        print(f"Počet prvků: {encoded.size}")
        return

    raise ValueError(f"Sekvence s indexem {index} nebyla nalezena v souboru {path}")


def step_5_build_model(path: str) -> VAE:
    """Step 4: build model architecture without training yet."""
    print_header("Krok 5: sestav model, ale ještě ho netrainuj")
    _, seq_len = get_fasta_file_dimensions(path)
    print(f"Délka sekvence pro model: {seq_len}")
    model = VAE(
        num_aa_types=23,
        dim_latent_vars=2,
        dim_msa_vars=seq_len,
        num_hidden_units=seq_len * 3,
        activation_func="relu",
        regularization=1e-4,
    )
    model.compute_output_shape(input_shape=(None, 23 * seq_len))
    print("Model byl vytvořen.")
    print("Encoder summary:")
    model.encoder.summary()
    print("Decoder summary:")
    model.decoder.summary()
    return model


def step_6_make_dataset(path: str, batch_size: int = 4) -> tf.data.Dataset:
    """Optional helper step: build a TensorFlow dataset and inspect one batch."""
    print_header("Krok 6: volitelný helper - dataset")
    ds = tf.data.Dataset.from_generator(
        lambda: read_fasta_as_one_hot_encoded(path),
        tf.int8,
    )
    ds = ds.shuffle(100)
    ds = ds.batch(batch_size)

    for batch in ds.take(1):
        print(f"Shape batch: {batch.shape}")
        print(f"dtype: {batch.dtype}")
        print(f"První 3 řádky první sekvence: {batch[0, :3]}")

    return ds


def step_7_train_full_model(path: str, model_out: str = "trained_vae_manual", epochs: int = 1000, batch_size: int = 16) -> VAE:
    """Training step: creates model, trains it and saves it to disk."""
    print_header("Krok 7: trénink modelu")
    _, seq_len = get_fasta_file_dimensions(path)
    model = VAE(
        num_aa_types=23,
        dim_latent_vars=2,
        dim_msa_vars=seq_len,
        num_hidden_units=seq_len * 3,
        activation_func="relu",
        regularization=1e-4,
    )
    model.compute_output_shape(input_shape=(None, 23 * seq_len))

    ds = tf.data.Dataset.from_generator(
        lambda: read_fasta_as_one_hot_encoded(path),
        tf.int8,
    )
    ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_out,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    history = model.fit(
        ds,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, checkpoint],
    )
    model.save(model_out)
    print(f"Model uložen do: {model_out}")
    print("Poslední záznam loss:")
    print(history.history["loss"])
    return model


def step_8_train_dca_model(fasta_path: str, model_out: str = "trained_dca_manual.pkl") -> object:
    """Fit the mean-field DCA model to a FASTA alignment and save it."""
    print_header("Krok 8: trénink DCA modelu")
    from dca.dca_class import dca

    dca_model = dca(fasta_path)
    dca_model.mean_field()
    output_path = Path(model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_handle:
        pickle.dump(dca_model, file_handle)
    print(f"Natrénovaný DCA model uložen do: {output_path}")
    return dca_model


def _load_trained_dca_model(path: str):
    """Load a DCA model saved by step 8."""
    with open(path, "rb") as file_handle:
        dca_model = pickle.load(file_handle)
    if (
        not callable(getattr(dca_model, "compute_Hamiltonian", None))
        or not hasattr(dca_model, "couplings")
        or not hasattr(dca_model, "localfields")
    ):
        raise ValueError(f"Soubor neobsahuje natrénovaný DCA model: {path}")
    return dca_model


def step_10_plot_latent_space(model_path: str, fasta_path: str, output_path: str = "latent_scatter.png") -> None:
    """Visualization step: encode sequences and plot their 2D latent coordinates."""
    print_header("Krok 10: zobrazení latentního prostoru")
    model = tf.keras.models.load_model(model_path, compile=False)
    ds = tf.data.Dataset.from_generator(
        lambda: read_fasta_as_one_hot_encoded(fasta_path),
        tf.int8,
    )
    ds = ds.batch(32)
    z_mean, _, _ = model.encoder.predict(ds)

    plt.figure(figsize=(7, 7))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=20, alpha=0.8)
    plt.title("Latent space")
    plt.xlabel("z0")
    plt.ylabel("z1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"Obrázek uložen do: {output_path}")


def step_11_plot_decoder_grid(model_path: str, output_path: str = "latent_decoder_map.png", grid_size: int = 25) -> None:
    """Visualization step: sample latent grid and show a decoder map."""
    print_header("Krok 11: mapa decoderu v latentním prostoru")
    model = tf.keras.models.load_model(model_path, compile=False)
    coords = np.linspace(-3, 3, grid_size)
    grid_x, grid_y = np.meshgrid(coords, coords)
    latent = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    decoded = model.decoder.predict(latent, verbose=0)
    scores = decoded.mean(axis=(1, 2))
    heatmap = scores.reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, origin="lower", cmap="viridis", extent=[-3, 3, -3, 3])
    plt.colorbar(label="decoded mean activity")
    plt.title("Decoder latent map")
    plt.xlabel("z0")
    plt.ylabel("z1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"Obrázek uložen do: {output_path}")


def step_1_load_existing_model_and_plot(
    model_path: str,
    fasta_path: str,
    output_path: str = "loaded_model_latent.png",
) -> VAE:
    """Load an already trained model and plot it without retraining."""
    print_header("Krok 1: načtení už natrénovaného modelu a vizualizace")
    trained_model = tf.keras.models.load_model(model_path, compile=False)
    ds = tf.data.Dataset.from_generator(
        lambda: read_fasta_as_one_hot_encoded(fasta_path),
        tf.int8,
    )
    ds = ds.batch(32)
    z_mean, _, _ = trained_model.encoder.predict(ds)

    plt.figure(figsize=(7, 7))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], s=20, alpha=0.9)
    plt.title(f"Loaded model: {model_path}")
    plt.xlabel("z0")
    plt.ylabel("z1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"Načtený model byl vizualizován a uložen do: {output_path}")
    return trained_model


def _get_axis_values(model_path: str, fasta_path: str) -> int:
    """Use the original application's latent extent calculation."""
    model = tf.keras.models.load_model(model_path, compile=False)
    dataset = tf.data.Dataset.from_generator(
        lambda: read_fasta_as_one_hot_encoded(fasta_path), tf.int8
    ).batch(1)
    z_mean, _, _ = model.encoder.predict(dataset, verbose=0)
    largest_value = abs(max(z_mean.min(), z_mean.max()))
    return int(np.ceil(largest_value))


def _make_grid_msa(model_path: str, output_path: str, dimension: int, resolution: int, batch_size: int = 10000) -> np.ndarray:
    """Copy the original application's decoder-grid-to-FASTA procedure."""
    model = tf.keras.models.load_model(model_path, compile=False)
    sampling_set = np.linspace(-dimension, dimension, resolution)
    mesh = np.meshgrid(sampling_set, sampling_set)
    coordinates = np.vstack(np.array(mesh).transpose())

    with open(output_path, "w") as file_handle:
        for batch_index in range(0, coordinates.shape[0], batch_size):
            latent_batch = coordinates[batch_index:batch_index + batch_size]
            latent_output = model.decoder.predict(latent_batch, verbose=0)
            sequences = []
            for sequence_matrix in latent_output:
                sequence = "".join(
                    next(key for key, value in seq_code.items() if value == index)
                    for index in np.argsort(sequence_matrix, axis=0)[-1, :]
                )
                sequences.append(sequence)
            for index, (x_value, y_value) in enumerate(latent_batch):
                file_handle.write(f"> {x_value} {y_value}\n{sequences[index]}\n")
    return coordinates


def step_9_generate_hamiltonian_map(
    model_path: str,
    fasta_path: str,
    output_path: Optional[str] = None,
    resolution: int = 500,
    dca_model_path: Optional[str] = None,
) -> str:
    """Generate the landscape using the DCA model saved by step 7."""
    print_header("Krok 9: generování originální Hamiltonian mapy")

    model_directory = Path(model_path)
    model_directory.mkdir(parents=True, exist_ok=True)
    landscape_path = model_directory / "temp_landscape.fasta"
    output_pickle = Path(output_path) if output_path else model_directory / "latent_generative_landscape.pkl"
    trained_dca_path = Path(dca_model_path) if dca_model_path else model_directory.parent / "trained_dca_manual.pkl"
    if not trained_dca_path.is_file():
        raise FileNotFoundError(
            f"Natrénovaný DCA model nebyl nalezen: {trained_dca_path}. "
            "Nejprve spusť krok 8 nebo nastav dca_model_path."
        )

    dimension = _get_axis_values(model_path, fasta_path)
    coordinates = _make_grid_msa(model_path, str(landscape_path), dimension, resolution)

    dca_model = _load_trained_dca_model(str(trained_dca_path))
    grid_hamiltonians, _ = dca_model.compute_Hamiltonian(str(landscape_path))
    output_grid = np.zeros((coordinates.shape[0], 3))
    output_grid[:, :2] = coordinates
    output_grid[:, 2] = grid_hamiltonians

    _, sequence_length = get_fasta_file_dimensions(fasta_path)
    with open(fasta_path, "rb") as file_handle:
        training_sequences = base64.b64encode(file_handle.read())
    with open(landscape_path, "rb") as file_handle:
        landscape_sequences = base64.b64encode(file_handle.read())
    model = tf.keras.models.load_model(model_path, compile=False)
    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with bz2.BZ2File(output_pickle, "wb") as file_handle:
        pickle.dump([
            str(model_path),
            output_grid,
            sequence_length,
            model.get_weights(),
            training_sequences,
            landscape_sequences,
        ], file_handle)
    landscape_path.unlink()
    print(f"Originální Hamiltonian mapa uložena do: {output_pickle}")
    return str(output_pickle)


def _load_original_landscape_payload(path: str) -> list:
    """Read the compressed payload produced by the original web application."""
    with bz2.BZ2File(path, "rb") as file_handle:
        payload = pickle.load(file_handle)
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Soubor není originální latent_generative_landscape.pkl.")
    return payload


def _load_original_landscape(path: str):
    """Read the grid from the compressed format produced by the original web application."""
    payload = _load_original_landscape_payload(path)
    grid = np.asarray(payload[1])
    if grid.ndim != 2 or grid.shape[1] < 3:
        raise ValueError("Landscape neobsahuje tabulku [x, y, Hamiltonian].")
    return grid


def _grid_arrays(grid: np.ndarray):
    """Match the original plot's x/y ordering and image orientation."""
    grid = grid[np.lexsort((grid[:, 1], grid[:, 0]))]
    pixels = grid[grid[:, 0] == grid[0, 0]].shape[0]
    image_grid = np.zeros((pixels, pixels))
    count = 0
    for i in range(pixels):
        for j in range(pixels):
            image_grid[j, i] = grid[count, 2]
            count += 1
    return grid, image_grid, pixels


def step_12_plot_dca_model(
    dca_pickle_path: str,
    output_path: str = "dca_map.png",
) -> None:
    """Display the DCA Hamiltonian map without VAE re-encoding overlays."""
    print_header("Krok 12: zobrazení DCA Hamiltonovy mapy")
    grid = _load_original_landscape(dca_pickle_path)
    grid, image_grid, _ = _grid_arrays(grid)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        image_grid,
        origin="lower",
        extent=[grid[0, 0], grid[-1, 0], grid[0, 1], grid[-1, 1]],
        cmap="viridis",
        aspect="auto",
    )
    figure.colorbar(image, ax=axis, label="Hamiltonian")
    axis.set(title="DCA Hamiltonian mapa", xlabel="z0", ylabel="z1")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"DCA Hamiltonova mapa uložena do: {output_path}")


def step_13_plot_dca_3d(dca_pickle_path: str, output_path: str = "dca_3d.html") -> None:
    """Visualization step: save the DCA/Hamiltonian landscape as an interactive 3D HTML."""
    print_header("Krok 13: interaktivní 3D DCA mapa")
    import plotly.graph_objects as go

    grid = _load_original_landscape(dca_pickle_path)
    grid, image_grid, pixels = _grid_arrays(grid)
    x_axis = np.linspace(grid[0, 0], grid[-1, 0], pixels)
    y_axis = np.linspace(grid[0, 1], grid[-1, 1], pixels)
    x, y = np.meshgrid(x_axis, y_axis)
    figure = go.Figure(data=[go.Surface(x=x, y=y, z=image_grid, colorscale="Viridis")])
    figure.update_layout(title="DCA/Hamiltonian landscape", scene=dict(xaxis_title="z0", yaxis_title="z1", zaxis_title="Hamiltonian"))
    figure.write_html(output_path)
    print(f"Interaktivní 3D mapa uložena do: {output_path}")


def _landscape_training_records(payload: list):
    """Load the original training FASTA records embedded in a landscape file."""
    if len(payload) < 5 or not isinstance(payload[4], (bytes, str)):
        raise ValueError("Landscape neobsahuje vložené trénovací FASTA sekvence.")
    fasta_data = base64.b64decode(payload[4]).decode("utf-8")
    records = list(SeqIO.parse(io.StringIO(fasta_data), "fasta"))
    if not records:
        raise ValueError("V landscape souboru nebyly nalezeny žádné trénovací sekvence.")
    return records


def _resolve_sequence_index(records, selector: str) -> int:
    """Resolve a FASTA header or zero-based sequence index."""
    matches = [
        index
        for index, record in enumerate(records)
        if selector == record.id or selector == record.description
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Výběr sekvence '{selector}' odpovídá více záznamům FASTA.")

    try:
        sequence_index = int(selector)
    except ValueError as error:
        raise ValueError(
            f"Sekvence '{selector}' nebyla nalezena jako ID/hlavička ani index."
        ) from error
    if sequence_index < 0 or sequence_index >= len(records):
        raise ValueError(
            f"Index sekvence {sequence_index} je mimo rozsah 0 až {len(records) - 1}."
        )
    return sequence_index


def _sequence_identity_percent(first: str, second: str) -> float:
    """Return exact positional identity for two aligned sequences."""
    if len(first) != len(second):
        raise ValueError("Porovnávané sekvence musí mít stejnou délku.")
    if not first:
        raise ValueError("Nelze porovnat prázdné sekvence.")
    return 100.0 * sum(a == b for a, b in zip(first, second)) / len(first)


def _encode_sequence_batch(
    sequences: list[str],
    sequence_names: Optional[list[str]] = None,
) -> np.ndarray:
    """Encode aligned amino-acid sequences using the project's 23-symbol alphabet."""
    if not sequences or not sequences[0]:
        raise ValueError("Trénovací FASTA sekvence jsou prázdné.")
    sequence_length = len(sequences[0])
    if any(len(sequence) != sequence_length for sequence in sequences):
        raise ValueError("Trénovací FASTA sekvence nemají stejnou délku.")
    if sequence_names is None:
        sequence_names = [str(index) for index in range(len(sequences))]
    if len(sequence_names) != len(sequences):
        raise ValueError("Počet názvů sekvencí neodpovídá jejich počtu.")

    encoded = np.zeros((len(sequences), 23, sequence_length), dtype=np.float32)
    for sequence_index, sequence in enumerate(sequences):
        for position, amino_acid in enumerate(sequence.upper()):
            code = seq_code.get(amino_acid)
            if code is None:
                raise ValueError(
                    f"Nepodporovaný znak '{amino_acid}' v sekvenci "
                    f"{sequence_names[sequence_index]} na pozici {position + 1}."
                )
            codes = [code] if isinstance(code, int) else list(code)
            if not codes or any(not isinstance(item, int) or not 0 <= item < 23 for item in codes):
                raise ValueError(
                    f"Neplatné kódování znaku '{amino_acid}' v sekvenci "
                    f"{sequence_names[sequence_index]} na pozici {position + 1}."
                )
            encoded[sequence_index, codes, position] = 1.0 / len(codes)
    return encoded


def _correlation_coefficients(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return Pearson/Spearman coefficients and p-values when they are defined."""
    from scipy.stats import pearsonr, spearmanr

    if first.size < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None, None, None, None
    pearson_correlation, pearson_p_value = pearsonr(first, second)
    spearman_correlation, spearman_p_value = spearmanr(first, second)
    return (
        float(pearson_correlation),
        float(pearson_p_value),
        float(spearman_correlation),
        float(spearman_p_value),
    )


def _format_correlation(value: Optional[float], p_value: Optional[float]) -> str:
    if value is None or p_value is None or not np.isfinite(value) or not np.isfinite(p_value):
        return "nedefinováno"
    return f"r={value:.3f}, p={p_value:.3g}"


def step_14_plot_dca_sequences(
    dca_pickle_path: str,
    vae_model,
    output_path: str = "dca_sequences.html",
    sequence: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    three_dimensional: bool = False,
) -> None:
    """Create an interactive 2D/3D DCA map with optional identity highlighting."""
    print_header("Krok 14: interaktivní DCA mapa sekvencí")
    import plotly.graph_objects as go

    if (sequence is None) != (similarity_threshold is None):
        raise ValueError(
            "Pro krok 14 zadej společně --sequence a --similarity-threshold."
        )
    if similarity_threshold is not None and not 0 <= similarity_threshold <= 100:
        raise ValueError("Hranice podobnosti pro krok 14 musí být v rozsahu 0 až 100 procent.")

    payload = _load_original_landscape_payload(dca_pickle_path)
    grid = np.asarray(payload[1])
    if grid.ndim != 2 or grid.shape[1] < 3:
        raise ValueError("Landscape neobsahuje tabulku [x, y, Hamiltonian].")
    grid, image_grid, pixels = _grid_arrays(grid)
    records = _landscape_training_records(payload)

    sequences = [str(record.seq).upper() for record in records]
    sequence_length = len(sequences[0])
    encoded_sequences = _encode_sequence_batch(
        sequences,
        [record.id for record in records],
    )

    latent_coordinates, _, _ = vae_model.encoder.predict(
        encoded_sequences.reshape(len(sequences), 23 * sequence_length),
        batch_size=2048,
        verbose=0,
    )
    if latent_coordinates.ndim != 2 or latent_coordinates.shape[1] < 2:
        raise ValueError("VAE encoder nevrátil alespoň dvě latentní souřadnice.")
    if latent_coordinates.shape[0] != len(records):
        raise ValueError("Počet latentních souřadnic neodpovídá počtu FASTA sekvencí.")

    selected_index = None
    identities = None
    similar_indices = set()
    if sequence is not None:
        selected_index = _resolve_sequence_index(records, sequence)
        selected_sequence = sequences[selected_index]
        identities = [
            _sequence_identity_percent(selected_sequence, item)
            for item in sequences
        ]
        similar_indices = {
            index
            for index, identity in enumerate(identities)
            if identity >= similarity_threshold and index != selected_index
        }

    x_axis = np.linspace(grid[0, 0], grid[-1, 0], pixels)
    y_axis = np.linspace(grid[0, 1], grid[-1, 1], pixels)
    figure = go.Figure()
    if three_dimensional:
        x_surface, y_surface = np.meshgrid(x_axis, y_axis)
        figure.add_trace(
            go.Surface(
                x=x_surface,
                y=y_surface,
                z=image_grid,
                colorscale="Viridis",
                colorbar=dict(title="Hamiltonian"),
                name="DCA landscape",
            )
        )
        x_upper = np.clip(
            np.searchsorted(x_axis, latent_coordinates[:, 0]), 0, pixels - 1
        )
        y_upper = np.clip(
            np.searchsorted(y_axis, latent_coordinates[:, 1]), 0, pixels - 1
        )
        x_lower = np.maximum(x_upper - 1, 0)
        y_lower = np.maximum(y_upper - 1, 0)
        x_indices = np.where(
            np.abs(x_axis[x_upper] - latent_coordinates[:, 0])
            < np.abs(x_axis[x_lower] - latent_coordinates[:, 0]),
            x_upper,
            x_lower,
        )
        y_indices = np.where(
            np.abs(y_axis[y_upper] - latent_coordinates[:, 1])
            < np.abs(y_axis[y_lower] - latent_coordinates[:, 1]),
            y_upper,
            y_lower,
        )
        sequence_heights = image_grid[y_indices, x_indices] + 0.4
    else:
        figure.add_trace(
            go.Heatmap(
                x=x_axis,
                y=y_axis,
                z=image_grid,
                colorscale="Viridis",
                colorbar=dict(title="Hamiltonian"),
                hovertemplate="z0: %{x:.3f}<br>z1: %{y:.3f}<br>Hamiltonian: %{z:.3f}<extra></extra>",
                name="DCA landscape",
            )
        )

    def add_sequence_trace(indices, name: str, color: str, size: int) -> None:
        if not indices:
            return
        customdata = [
            [
                records[index].description,
                sequences[index],
                f"{identities[index]:.1f}%" if identities is not None else "N/A",
            ]
            for index in indices
        ]
        marker = dict(color=color, size=size, line=dict(width=0.5, color="white"))
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Sekvence: %{customdata[1]}<br>"
            "Identita vůči vybrané: %{customdata[2]}<br>"
            "z0: %{x:.3f}<br>z1: %{y:.3f}"
        )
        if three_dimensional:
            figure.add_trace(
                go.Scatter3d(
                    x=latent_coordinates[indices, 0],
                    y=latent_coordinates[indices, 1],
                    z=sequence_heights[indices],
                    mode="markers",
                    name=name,
                    customdata=customdata,
                    marker=marker,
                    hovertemplate=hovertemplate + "<br>Hamiltonian: %{z:.3f}<extra></extra>",
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=latent_coordinates[indices, 0],
                    y=latent_coordinates[indices, 1],
                    mode="markers",
                    name=name,
                    customdata=customdata,
                    marker=marker,
                    hovertemplate=hovertemplate + "<extra></extra>",
                )
            )

    if selected_index is None:
        add_sequence_trace(list(range(len(records))), "Sekvence", "#ffffff", 7)
        title = "DCA landscape se sekvencemi"
    else:
        base_indices = [
            index
            for index in range(len(records))
            if index != selected_index and index not in similar_indices
        ]
        add_sequence_trace(base_indices, "Ostatní sekvence", "#1f77b4", 6)
        add_sequence_trace(sorted(similar_indices), "Podobné sekvence", "#00e5ff", 9)
        add_sequence_trace([selected_index], "Vybraná sekvence", "#ff3030", 13)
        title = (
            f"DCA landscape – {records[selected_index].id}; "
            f"identita ≥ {similarity_threshold:g}%"
        )

    layout = dict(
        title=title,
        template="plotly_dark",
        height=800,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=55),
    )
    if three_dimensional:
        layout["scene"] = dict(
            xaxis_title="z0",
            yaxis_title="z1",
            zaxis_title="Hamiltonian",
        )
    else:
        layout["xaxis_title"] = "z0"
        layout["yaxis_title"] = "z1"
        layout["xaxis"] = dict(scaleanchor="y", scaleratio=1)
    figure.update_layout(**layout)
    figure.write_html(output_path, include_plotlyjs=True)
    dimension = "3D" if three_dimensional else "2D"
    print(f"Interaktivní {dimension} mapa sekvencí uložena do: {output_path}")
    if selected_index is not None:
        print(
            f"Vybraná sekvence: {records[selected_index].description}; "
            f"podobných sekvencí nad hranicí: {len(similar_indices)}"
        )


def step_15_evaluate_dca_reconstruction(
    dca_pickle_path: str,
    vae_model,
    csv_output_path: str = "dca_reconstruction_report.csv",
    plot_output_path: str = "dca_reconstruction_correlation.html",
) -> None:
    """Compare decoder reconstructions with training sequences and their DCA scores."""
    print_header("Krok 15: kvalita rekonstrukce sekvencí a korelace s DCA")
    from plotly import graph_objects as go
    from scipy.spatial import cKDTree

    payload = _load_original_landscape_payload(dca_pickle_path)
    grid = np.asarray(payload[1], dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] < 3 or grid.shape[0] == 0:
        raise ValueError("Landscape neobsahuje tabulku [x, y, Hamiltonian].")
    if not np.isfinite(grid[:, :3]).all():
        raise ValueError("Landscape obsahuje neplatné souřadnice nebo Hamiltonian skóre.")

    records = _landscape_training_records(payload)
    sequences = [str(record.seq).upper() for record in records]
    encoded_sequences = _encode_sequence_batch(
        sequences,
        [record.id for record in records],
    )
    model_inputs = encoded_sequences.reshape(len(sequences), -1)
    original_latent, _, _ = vae_model.encoder.predict(
        model_inputs,
        batch_size=2048,
        verbose=0,
    )
    if (
        original_latent.ndim != 2
        or original_latent.shape[0] != len(records)
        or original_latent.shape[1] < 2
    ):
        raise ValueError("VAE encoder nevrátil platné latentní souřadnice.")

    decoder_probabilities = vae_model.decoder.predict(
        original_latent,
        batch_size=2048,
        verbose=0,
    )
    if (
        decoder_probabilities.ndim != 3
        or decoder_probabilities.shape[0] != len(records)
        or decoder_probabilities.shape[1] != 23
        or decoder_probabilities.shape[2] != encoded_sequences.shape[2]
    ):
        raise ValueError("VAE decoder nevrátil výstup se správným tvarem sekvencí.")
    if not np.isfinite(original_latent).all() or not np.isfinite(decoder_probabilities).all():
        raise ValueError("VAE encoder nebo decoder vrátil neplatné hodnoty.")
    if (
        not np.issubdtype(decoder_probabilities.dtype, np.floating)
        or np.any(decoder_probabilities < 0)
        or not np.allclose(decoder_probabilities.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError("Výstup VAE decoderu neobsahuje platná pravděpodobnostní rozdělení.")

    amino_acids_by_code = {
        code: amino_acid
        for amino_acid, code in seq_code.items()
        if isinstance(code, int)
    }
    if len(amino_acids_by_code) != 23:
        raise ValueError("Abeceda sekvencí neobsahuje přesně 23 aminokyselinových symbolů.")
    reconstructed_sequences = [
        "".join(amino_acids_by_code[code] for code in sequence_codes)
        for sequence_codes in np.argmax(decoder_probabilities, axis=1)
    ]
    identity_percentages = np.asarray(
        [
            _sequence_identity_percent(original, reconstructed)
            for original, reconstructed in zip(sequences, reconstructed_sequences)
        ],
        dtype=np.float64,
    )
    mutation_percentages = 100.0 - identity_percentages
    decoder_confidence_percentages = (
        np.max(decoder_probabilities, axis=1).mean(axis=1) * 100.0
    )
    clipped_probabilities = np.clip(
        decoder_probabilities,
        np.finfo(decoder_probabilities.dtype).tiny,
        1.0,
    )
    decoder_uncertainty_percentages = (
        -np.sum(
            clipped_probabilities * np.log(clipped_probabilities),
            axis=1,
            dtype=np.float64,
        ).mean(axis=1)
        / np.log(23)
        * 100.0
    )

    reconstructed_encoded = _encode_sequence_batch(
        reconstructed_sequences,
        [record.id for record in records],
    ).reshape(len(records), -1)
    reconstructed_latent, _, _ = vae_model.encoder.predict(
        reconstructed_encoded,
        batch_size=2048,
        verbose=0,
    )
    if (
        reconstructed_latent.ndim != 2
        or reconstructed_latent.shape[0] != len(records)
        or reconstructed_latent.shape[1] < 2
        or not np.isfinite(reconstructed_latent).all()
    ):
        raise ValueError("VAE encoder nevrátil platné souřadnice rekonstruovaných sekvencí.")

    landscape_tree = cKDTree(grid[:, :2])
    original_distances, original_grid_indices = landscape_tree.query(
        original_latent[:, :2],
        k=1,
    )
    reconstructed_distances, reconstructed_grid_indices = landscape_tree.query(
        reconstructed_latent[:, :2],
        k=1,
    )
    original_dca_scores = grid[original_grid_indices, 2]
    reconstructed_dca_scores = grid[reconstructed_grid_indices, 2]
    dca_score_changes = reconstructed_dca_scores - original_dca_scores

    csv_path = Path(csv_output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "description",
        "original_sequence",
        "reconstructed_sequence",
        "identity_percent",
        "changed_percent",
        "decoder_mean_confidence_percent",
        "decoder_normalized_entropy_percent",
        "original_z0",
        "original_z1",
        "original_dca_hamiltonian",
        "original_nearest_grid_distance",
        "reconstructed_z0",
        "reconstructed_z1",
        "reconstructed_dca_hamiltonian",
        "reconstructed_nearest_grid_distance",
        "dca_hamiltonian_change",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "sequence_id": record.id,
                    "description": record.description,
                    "original_sequence": sequences[index],
                    "reconstructed_sequence": reconstructed_sequences[index],
                    "identity_percent": f"{identity_percentages[index]:.4f}",
                    "changed_percent": f"{mutation_percentages[index]:.4f}",
                    "decoder_mean_confidence_percent": f"{decoder_confidence_percentages[index]:.4f}",
                    "decoder_normalized_entropy_percent": f"{decoder_uncertainty_percentages[index]:.4f}",
                    "original_z0": f"{original_latent[index, 0]:.8f}",
                    "original_z1": f"{original_latent[index, 1]:.8f}",
                    "original_dca_hamiltonian": f"{original_dca_scores[index]:.8f}",
                    "original_nearest_grid_distance": f"{original_distances[index]:.8f}",
                    "reconstructed_z0": f"{reconstructed_latent[index, 0]:.8f}",
                    "reconstructed_z1": f"{reconstructed_latent[index, 1]:.8f}",
                    "reconstructed_dca_hamiltonian": f"{reconstructed_dca_scores[index]:.8f}",
                    "reconstructed_nearest_grid_distance": f"{reconstructed_distances[index]:.8f}",
                    "dca_hamiltonian_change": f"{dca_score_changes[index]:.8f}",
                }
            )

    best_index = int(np.argmax(identity_percentages))
    worst_index = int(np.argmin(identity_percentages))
    noisiest_index = int(np.argmax(decoder_uncertainty_percentages))
    print(f"Vyhodnoceno sekvencí: {len(records)}")
    print(
        "Shoda rekonstrukcí: "
        f"průměr {np.mean(identity_percentages):.2f} %, "
        f"medián {np.median(identity_percentages):.2f} %"
    )
    print(
        "Změněné pozice: "
        f"průměr {np.mean(mutation_percentages):.2f} %, "
        f"medián {np.median(mutation_percentages):.2f} %"
    )
    print(
        "Nejistota decoderu (normalizovaná entropie): "
        f"průměr {np.mean(decoder_uncertainty_percentages):.2f} %, "
        f"medián {np.median(decoder_uncertainty_percentages):.2f} %; "
        f"průměrná jistota {np.mean(decoder_confidence_percentages):.2f} %"
    )
    print(
        f"Nejvíce změněná sekvence: {records[worst_index].description}; "
        f"shoda {identity_percentages[worst_index]:.2f} %, "
        f"změněno {mutation_percentages[worst_index]:.2f} %"
    )
    print(
        f"Nejvyšší shoda: {records[best_index].description}; "
        f"{identity_percentages[best_index]:.2f} %"
    )
    print(
        f"Nejistější rekonstrukce: {records[noisiest_index].description}; "
        f"entropie {decoder_uncertainty_percentages[noisiest_index]:.2f} %, "
        f"průměrná jistota {decoder_confidence_percentages[noisiest_index]:.2f} %"
    )

    correlations = {
        "původní sekvence": _correlation_coefficients(
            identity_percentages,
            original_dca_scores,
        ),
        "rekonstruovaná sekvence": _correlation_coefficients(
            identity_percentages,
            reconstructed_dca_scores,
        ),
        "změna skóre (rekonstrukce - původní)": _correlation_coefficients(
            identity_percentages,
            dca_score_changes,
        ),
    }
    for label, (pearson, pearson_p, spearman, spearman_p) in correlations.items():
        print(
            f"Korelace shody s DCA Hamiltonianem ({label}): "
            f"Pearson {_format_correlation(pearson, pearson_p)}; "
            f"Spearman {_format_correlation(spearman, spearman_p)}"
        )
    print("DCA Hamiltonian je energetické skóre landscape, nikoli míra entropie.")

    figure = go.Figure()
    for label, scores, color in (
        ("Původní sekvence", original_dca_scores, "#1f77b4"),
        ("Rekonstruovaná sekvence", reconstructed_dca_scores, "#ff7f0e"),
    ):
        figure.add_trace(
            go.Scatter(
                x=identity_percentages,
                y=scores,
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.75),
                customdata=[
                    [record.id, record.description]
                    for record in records
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Shoda rekonstrukce: %{x:.2f} %<br>"
                    "DCA Hamiltonian: %{y:.4f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Podobnost rekonstrukce sekvence vs. DCA Hamiltonian",
        xaxis_title="Shoda sekvence po rekonstrukci (%)",
        yaxis_title="DCA Hamiltonian (nejbližší bod landscape)",
        template="plotly_dark",
        height=750,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plot_path = Path(plot_output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(plot_path), include_plotlyjs=True)
    print(f"Detailní výsledky uloženy do: {csv_path}")
    print(f"Korelační graf uložen do: {plot_path}")


def run_steps(
    steps: list[int],
    fasta_path: str = DEFAULT_FASTA,
    model_path: str = "trained_vae_manual",
    epochs: int = 1000,
    batch_size: int = 16,
    dca_pickle_path: Optional[str] = None,
    dca_model_path: Optional[str] = None,
    sequence: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    three_dimensional: bool = False,
) -> None:
    """Run selected steps directly from Python, without command-line parsing."""
    trained_model: Optional[VAE] = None
    trained_dca_path = dca_model_path or str(Path(model_path).parent / "trained_dca_manual.pkl")
    for step_num in steps:
        if step_num == 1:
            trained_model = step_1_load_existing_model_and_plot(
                model_path,
                fasta_path,
                str(Path(model_path) / "loaded_model_latent.png"),
            )
        elif step_num == 2:
            step_2_inspect_fasta(fasta_path)
        elif step_num == 3:
            step_3_preview_sequences(fasta_path, limit=2)
        elif step_num == 4:
            step_4_encode_one_sequence(fasta_path, index=0)
        elif step_num == 5:
            step_5_build_model(fasta_path)
        elif step_num == 6:
            step_6_make_dataset(fasta_path, batch_size=4)
        elif step_num == 7:
            trained_model = step_7_train_full_model(
                fasta_path,
                model_out=model_path,
                epochs=epochs,
                batch_size=batch_size,
            )
        elif step_num == 8:
            step_8_train_dca_model(fasta_path, model_out=trained_dca_path)
        elif step_num == 9:
            landscape_path = dca_pickle_path or str(Path(model_path) / "latent_generative_landscape.pkl")
            step_9_generate_hamiltonian_map(
                model_path,
                fasta_path,
                output_path=landscape_path,
                dca_model_path=trained_dca_path,
            )
        elif step_num == 10:
            step_10_plot_latent_space(
                model_path, fasta_path, str(Path(model_path) / "latent_scatter.png")
            )
        elif step_num == 11:
            step_11_plot_decoder_grid(
                model_path, str(Path(model_path) / "latent_decoder_map.png")
            )
        elif step_num == 12:
            landscape_path = dca_pickle_path or str(Path(model_path) / "latent_generative_landscape.pkl")
            step_12_plot_dca_model(
                landscape_path,
                str(Path(model_path) / "dca_map.png"),
            )
        elif step_num == 13:
            landscape_path = dca_pickle_path or str(Path(model_path) / "latent_generative_landscape.pkl")
            step_13_plot_dca_3d(landscape_path, str(Path(model_path) / "dca_3d.html"))
        elif step_num == 14:
            landscape_path = dca_pickle_path or str(Path(model_path) / "latent_generative_landscape.pkl")
            if trained_model is None:
                trained_model = tf.keras.models.load_model(model_path, compile=False)
            step_14_plot_dca_sequences(
                landscape_path,
                trained_model,
                str(Path(model_path) / "dca_sequences.html"),
                sequence=sequence,
                similarity_threshold=similarity_threshold,
                three_dimensional=three_dimensional,
            )
        elif step_num == 15:
            landscape_path = dca_pickle_path or str(Path(model_path) / "latent_generative_landscape.pkl")
            if trained_model is None:
                trained_model = tf.keras.models.load_model(model_path, compile=False)
            step_15_evaluate_dca_reconstruction(
                landscape_path,
                trained_model,
                str(Path(model_path) / "dca_reconstruction_report.csv"),
                str(Path(model_path) / "dca_reconstruction_correlation.html"),
            )
        else:
            raise ValueError(
                f"Neplatný krok {step_num}. Povolené kroky jsou 1 až 15."
            )


def run_all_steps(
    path: str,
    model_out: str = "trained_vae_manual",
    epochs: int = 1000,
    batch_size: int = 16,
    dca_model_out: Optional[str] = None,
) -> None:
    """Run the complete workflow, including original Hamiltonian visualizations."""
    run_steps(
        list(range(1, 16)),
        fasta_path=path,
        model_path=model_out,
        epochs=epochs,
        batch_size=batch_size,
        dca_model_path=dca_model_out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn LGL-VAE step-by-step")
    parser.add_argument("--path", default=DEFAULT_FASTA, help="Cesta k FASTA souboru")
    parser.add_argument(
        "--step",
        nargs="+",
        default=["0"],
        help="Kroky ke spuštění, např. 1 2 3 4 nebo 1,2,3,4 nebo all",
    )
    parser.add_argument(
        "--model-out",
        default="trained_vae_manual",
        help="Výstupní cesta pro uložení modelu po tréninku",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Počet epoch pro trénink (výchozí: 1000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Velikost batch_size pro trénink (výchozí: 16)")
    parser.add_argument(
        "--dca-model-out",
        default=None,
        help="Cesta pro uložení/načtení natrénovaného DCA modelu (výchozí: vedle modelu VAE)",
    )
    parser.add_argument("--dca-pkl", default=None, help="Volitelná cesta k latent_generative_landscape.pkl")
    parser.add_argument(
        "--sequence",
        default=None,
        help="Pro krok 14: ID/hlavička FASTA sekvence nebo její index od nuly",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Pro krok 14: minimální sekvenční identita v procentech (0–100)",
    )
    parser.add_argument(
        "--three-dimensional",
        action="store_true",
        help="Pro krok 14: vykreslit DCA mapu se sekvencemi ve 3D",
    )
    args = parser.parse_args()

    if not Path(args.path).exists():
        raise FileNotFoundError(f"Soubor nebyl nalezen: {args.path}")

    steps = parse_steps(args.step)
    if not steps:
        print("Vyber krok: 1 až 15 nebo all")
        return

    run_steps(
        [step for step in steps if step != 0],
        fasta_path=args.path,
        model_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dca_pickle_path=args.dca_pkl,
        dca_model_path=args.dca_model_out,
        sequence=args.sequence,
        similarity_threshold=args.similarity_threshold,
        three_dimensional=args.three_dimensional,
    )


if __name__ == "__main__":
    main()
