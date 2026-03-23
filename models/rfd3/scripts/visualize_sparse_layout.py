from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import torch


DEFAULT_ATTENTION_CELL_SIZE = 4
DEFAULT_PAIRWISE_CELL_SIZE = 1
BLOCK_SIZE = 16
PADDING = 24
GAP = 32
TEXT_SCALE = 2
LABEL_BAND = 28

BACKGROUND = (255, 253, 248)
PANEL = (244, 241, 232)
BORDER = (200, 195, 182)
TEXT = (16, 32, 48)
LOCAL = (198, 93, 38)
NONLOCAL = (11, 110, 79)
BLOCK = (42, 83, 160)
PAIR = (11, 110, 79)
MOTIF = (198, 93, 38)
REFERENCE = (42, 83, 160)

FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "[": ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    "]": ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "00100"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
}


def _hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _load_payload(path: Path) -> tuple[str, dict[str, torch.Tensor] | torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "indices" in payload:
        return "attention", payload
    if isinstance(payload, dict) and "pair_nonzero_mask" in payload:
        return "pairwise", payload
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 3:
            return "attention", {"indices": payload}
        raise ValueError(f"Unsupported tensor payload with shape {tuple(payload.shape)}")
    raise ValueError(f"Unsupported payload format in {path}")


def _load_indices(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    indices = payload["indices"]
    if not isinstance(indices, torch.Tensor) or indices.ndim != 3:
        raise ValueError("Expected indices tensor with shape [B, L, k]")
    return indices.to(dtype=torch.long, device="cpu")


def _squeeze_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool, device="cpu")


def _attention_views(
    indices: torch.Tensor,
    batch_index: int,
    block_size: int,
    local_radius: int,
) -> dict[str, torch.Tensor | int | float]:
    if batch_index < 0 or batch_index >= indices.shape[0]:
        raise IndexError(
            f"batch_index={batch_index} out of range for batch size {indices.shape[0]}"
        )
    batch = indices[batch_index]
    length = batch.shape[0]
    adjacency = torch.zeros((length, length), dtype=torch.bool)
    rows = torch.arange(length).unsqueeze(1).expand_as(batch)
    valid = (batch >= 0) & (batch < length)
    adjacency[rows[valid], batch[valid]] = True
    row_ids = torch.arange(length)
    distance = torch.abs(row_ids[:, None] - row_ids[None, :])
    local_mask = distance <= local_radius
    local_edges = adjacency & local_mask
    nonlocal_edges = adjacency & ~local_mask
    occupancy = _block_occupancy(adjacency, block_size=block_size)
    return {
        "adjacency": adjacency,
        "local_edges": local_edges,
        "nonlocal_edges": nonlocal_edges,
        "occupancy": occupancy,
        "length": length,
        "k": int(batch.shape[1]),
        "nnz": int(adjacency.sum().item()),
        "density": float(adjacency.float().mean().item()),
        "local_radius": local_radius,
    }


def _pairwise_views(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | int | float]:
    pair_nonzero = _squeeze_mask(payload["pair_nonzero_mask"])
    motif = _squeeze_mask(payload["motif_valid_mask"])
    reference = _squeeze_mask(payload["ref_valid_mask"])
    pair_energy = payload["pair_energy"].to(dtype=torch.float32, device="cpu")
    return {
        "pair_nonzero": pair_nonzero,
        "motif": motif,
        "reference": reference,
        "energy_nonzero": pair_energy > 0,
        "length": int(pair_nonzero.shape[0]),
        "pair_density": float(pair_nonzero.float().mean().item()),
        "motif_density": float(motif.float().mean().item()),
        "reference_density": float(reference.float().mean().item()),
    }


def _block_occupancy(adjacency: torch.Tensor, block_size: int) -> torch.Tensor:
    n_rows, n_cols = adjacency.shape
    row_blocks = (n_rows + block_size - 1) // block_size
    col_blocks = (n_cols + block_size - 1) // block_size
    occupancy = torch.zeros((row_blocks, col_blocks), dtype=torch.bool)
    for row_block in range(row_blocks):
        row_start = row_block * block_size
        row_end = min(row_start + block_size, n_rows)
        for col_block in range(col_blocks):
            col_start = col_block * block_size
            col_end = min(col_start + block_size, n_cols)
            occupancy[row_block, col_block] = adjacency[
                row_start:row_end, col_start:col_end
            ].any()
    return occupancy


def _row_offsets(adjacency: torch.Tensor) -> list[int]:
    counts = adjacency.sum(dim=1, dtype=torch.int32)
    offsets = [0]
    running = 0
    for count in counts.tolist():
        running += int(count)
        offsets.append(running)
    return offsets


def _svg_rect(x: int, y: int, width: int, height: int, fill: str, stroke: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1" />'
    )


def _svg_text(x: int, y: int, text: str, size: int = 14, weight: str = "400") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="monospace" font-size="{size}" '
        f'font-weight="{weight}" fill="#102030">{text}</text>'
    )


def _draw_matrix_svg(
    title: str,
    layers: list[tuple[torch.Tensor, tuple[int, int, int]]],
    origin_x: int,
    origin_y: int,
    cell_size: int,
) -> list[str]:
    matrix = layers[0][0]
    n_rows, n_cols = matrix.shape
    width = n_cols * cell_size
    height = n_rows * cell_size
    parts = [_svg_text(origin_x, origin_y - 8, title, size=15, weight="600")]
    parts.append(_svg_rect(origin_x, origin_y, width, height, _hex(PANEL), _hex(BORDER)))
    for layer, color in layers:
        for row, col in torch.nonzero(layer, as_tuple=False).tolist():
            x = origin_x + col * cell_size
            y = origin_y + row * cell_size
            parts.append(_svg_rect(x, y, cell_size, cell_size, _hex(color), _hex(color)))
    return parts


def _image_rect(
    image: bytearray,
    image_width: int,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    if width <= 0 or height <= 0:
        return
    for row in range(y, y + height):
        base = (row * image_width + x) * 3
        for _ in range(width):
            image[base] = color[0]
            image[base + 1] = color[1]
            image[base + 2] = color[2]
            base += 3


def _draw_text_png(
    image: bytearray,
    image_width: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = TEXT,
    scale: int = TEXT_SCALE,
) -> None:
    cursor_x = x
    for char in text.upper():
        glyph = FONT_5X7.get(char, FONT_5X7["?"])
        for row_index, row in enumerate(glyph):
            for col_index, bit in enumerate(row):
                if bit == "1":
                    _image_rect(
                        image,
                        image_width,
                        cursor_x + col_index * scale,
                        y + row_index * scale,
                        scale,
                        scale,
                        color,
                    )
        cursor_x += (len(glyph[0]) + 1) * scale


def _draw_matrix_png(
    image: bytearray,
    image_width: int,
    title: str,
    layers: list[tuple[torch.Tensor, tuple[int, int, int]]],
    origin_x: int,
    origin_y: int,
    cell_size: int,
) -> None:
    matrix = layers[0][0]
    n_rows, n_cols = matrix.shape
    width = n_cols * cell_size
    height = n_rows * cell_size
    _draw_text_png(image, image_width, origin_x, origin_y - LABEL_BAND + 4, title)
    _draw_text_png(image, image_width, origin_x, origin_y - LABEL_BAND + 18, "KEY")
    _draw_text_png(image, image_width, origin_x + width + 8, origin_y + 4, "QUERY")
    _image_rect(image, image_width, origin_x, origin_y, width, height, PANEL)
    _image_rect(image, image_width, origin_x, origin_y, width, 1, BORDER)
    _image_rect(image, image_width, origin_x, origin_y + height - 1, width, 1, BORDER)
    _image_rect(image, image_width, origin_x, origin_y, 1, height, BORDER)
    _image_rect(image, image_width, origin_x + width - 1, origin_y, 1, height, BORDER)
    for layer, color in layers:
        for row, col in torch.nonzero(layer, as_tuple=False).tolist():
            x = origin_x + col * cell_size
            y = origin_y + row * cell_size
            _image_rect(image, image_width, x, y, cell_size, cell_size, color)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc & 0xFFFFFFFF)


def _write_png(output_path: Path, image: bytearray, width: int, height: int) -> None:
    rows = []
    row_bytes = width * 3
    for row in range(height):
        start = row * row_bytes
        rows.append(b"\x00" + bytes(image[start : start + row_bytes]))
    compressed = zlib.compress(b"".join(rows), level=9)
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(
        _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
    )
    png.extend(_png_chunk(b"IDAT", compressed))
    png.extend(_png_chunk(b"IEND", b""))
    output_path.write_bytes(png)


def _attention_svg(
    views: dict[str, torch.Tensor | int | float],
    output_path: Path,
    block_size: int,
    cell_size: int,
) -> None:
    adjacency = views["adjacency"]
    local_edges = views["local_edges"]
    nonlocal_edges = views["nonlocal_edges"]
    occupancy = views["occupancy"]
    matrix_width = adjacency.shape[1] * cell_size
    matrix_height = adjacency.shape[0] * cell_size
    occupancy_width = occupancy.shape[1] * cell_size
    left_x = PADDING
    top_y = 56
    summary_x = left_x + max(matrix_width, occupancy_width) + GAP
    width = summary_x + 420
    height = max(top_y + matrix_height + GAP + occupancy.shape[0] * cell_size, 320) + PADDING
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, _hex(BACKGROUND), _hex(BACKGROUND)),
        _svg_text(PADDING, 28, "RFD3 Attention Layout View", size=18, weight="700"),
    ]
    parts.extend(
        _draw_matrix_svg(
            "Attention adjacency (local orange, nonlocal green)",
            [(nonlocal_edges, NONLOCAL), (local_edges, LOCAL)],
            left_x,
            top_y,
            cell_size,
        )
    )
    parts.extend(
        _draw_matrix_svg(
            "Block occupancy",
            [(occupancy, BLOCK)],
            left_x,
            top_y + matrix_height + GAP,
            cell_size,
        )
    )
    lines = [
        f"rows={adjacency.shape[0]} cols={adjacency.shape[1]}",
        f"k={views['k']} nnz={views['nnz']}",
        f"density={views['density']:.6f}",
        f"local_radius={views['local_radius']}",
        f"block_size={block_size}",
        f"row_offsets_head={_row_offsets(adjacency)[:8]}",
    ]
    for index, line in enumerate(lines):
        parts.append(_svg_text(summary_x, top_y + 20 * index, line, size=14))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _attention_png(
    views: dict[str, torch.Tensor | int | float],
    output_path: Path,
    block_size: int,
    cell_size: int,
) -> None:
    adjacency = views["adjacency"]
    local_edges = views["local_edges"]
    nonlocal_edges = views["nonlocal_edges"]
    occupancy = views["occupancy"]
    matrix_width = adjacency.shape[1] * cell_size
    matrix_height = adjacency.shape[0] * cell_size
    occupancy_width = occupancy.shape[1] * cell_size
    left_x = PADDING
    top_y = 56
    summary_x = left_x + max(matrix_width, occupancy_width) + GAP
    width = summary_x + 420
    height = max(top_y + matrix_height + GAP + occupancy.shape[0] * cell_size, 320) + PADDING
    image = bytearray(BACKGROUND * (width * height))
    _draw_text_png(image, width, PADDING, 20, "RFD3 ATTENTION LAYOUT VIEW")
    _draw_text_png(image, width, summary_x, 20, "LOCAL", LOCAL)
    _draw_text_png(image, width, summary_x + 100, 20, "NONLOCAL", NONLOCAL)
    _draw_text_png(image, width, summary_x + 250, 20, "BLOCK", BLOCK)
    _draw_matrix_png(
        image,
        width,
        "ATTENTION ADJACENCY",
        [(nonlocal_edges, NONLOCAL), (local_edges, LOCAL)],
        left_x,
        top_y,
        cell_size,
    )
    _draw_matrix_png(
        image,
        width,
        "BLOCK OCCUPANCY",
        [(occupancy, BLOCK)],
        left_x,
        top_y + matrix_height + GAP,
        cell_size,
    )
    lines = [
        f"ROWS={adjacency.shape[0]} COLS={adjacency.shape[1]}",
        f"K={views['k']} NNZ={views['nnz']}",
        f"DENSITY={views['density']:.6f}",
        f"LOCAL_RADIUS={views['local_radius']}",
        f"BLOCK_SIZE={block_size}",
    ]
    for index, line in enumerate(lines):
        _draw_text_png(image, width, summary_x, 64 + index * 24, line)
    _write_png(output_path, image, width, height)


def _pairwise_svg(
    views: dict[str, torch.Tensor | int | float],
    output_path: Path,
    cell_size: int,
) -> None:
    pair_nonzero = views["pair_nonzero"]
    motif = views["motif"]
    reference = views["reference"]
    matrix_width = pair_nonzero.shape[1] * cell_size
    matrix_height = pair_nonzero.shape[0] * cell_size
    left_x = PADDING
    top_y = 56
    width = left_x + matrix_width + 360
    height = top_y + 3 * matrix_height + 2 * GAP + PADDING
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, _hex(BACKGROUND), _hex(BACKGROUND)),
        _svg_text(PADDING, 28, "RFD3 Pairwise Initializer View", size=18, weight="700"),
    ]
    parts.extend(_draw_matrix_svg("Pair nonzero mask", [(pair_nonzero, PAIR)], left_x, top_y, cell_size))
    parts.extend(_draw_matrix_svg("Motif valid mask", [(motif, MOTIF)], left_x, top_y + matrix_height + GAP, cell_size))
    parts.extend(_draw_matrix_svg("Reference valid mask", [(reference, REFERENCE)], left_x, top_y + 2 * (matrix_height + GAP), cell_size))
    lines = [
        f"length={views['length']}",
        f"pair_density={views['pair_density']:.6f}",
        f"motif_density={views['motif_density']:.6f}",
        f"reference_density={views['reference_density']:.6f}",
    ]
    for index, line in enumerate(lines):
        parts.append(_svg_text(left_x + matrix_width + GAP, top_y + 20 * index, line, size=14))
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _pairwise_png(
    views: dict[str, torch.Tensor | int | float],
    output_path: Path,
    cell_size: int,
) -> None:
    pair_nonzero = views["pair_nonzero"]
    motif = views["motif"]
    reference = views["reference"]
    matrix_width = pair_nonzero.shape[1] * cell_size
    matrix_height = pair_nonzero.shape[0] * cell_size
    left_x = PADDING
    top_y = 56
    summary_x = left_x + matrix_width + GAP
    width = summary_x + 360
    height = top_y + 3 * matrix_height + 2 * GAP + PADDING
    image = bytearray(BACKGROUND * (width * height))
    _draw_text_png(image, width, PADDING, 20, "RFD3 PAIRWISE INITIALIZER VIEW")
    _draw_matrix_png(image, width, "PAIR NONZERO MASK", [(pair_nonzero, PAIR)], left_x, top_y, cell_size)
    _draw_matrix_png(image, width, "MOTIF VALID MASK", [(motif, MOTIF)], left_x, top_y + matrix_height + GAP, cell_size)
    _draw_matrix_png(image, width, "REFERENCE VALID MASK", [(reference, REFERENCE)], left_x, top_y + 2 * (matrix_height + GAP), cell_size)
    lines = [
        f"LENGTH={views['length']}",
        f"PAIR_DENSITY={views['pair_density']:.6f}",
        f"MOTIF_DENSITY={views['motif_density']:.6f}",
        f"REFERENCE_DENSITY={views['reference_density']:.6f}",
    ]
    for index, line in enumerate(lines):
        _draw_text_png(image, width, summary_x, 64 + index * 24, line)
    _write_png(output_path, image, width, height)


def render(
    kind: str,
    payload: dict[str, torch.Tensor] | torch.Tensor,
    output_path: Path,
    batch_index: int,
    block_size: int,
    local_radius: int,
    attention_cell_size: int,
    pairwise_cell_size: int,
) -> None:
    suffix = output_path.suffix.lower()
    if suffix not in {".svg", ".png"}:
        raise ValueError(f"Unsupported output format '{output_path.suffix}'. Use .svg or .png")

    if kind == "attention":
        views = _attention_views(
            _load_indices(payload),
            batch_index=batch_index,
            block_size=block_size,
            local_radius=local_radius,
        )
        if suffix == ".svg":
            _attention_svg(views, output_path, block_size=block_size, cell_size=attention_cell_size)
        else:
            _attention_png(views, output_path, block_size=block_size, cell_size=attention_cell_size)
        return

    if kind == "pairwise":
        views = _pairwise_views(payload)
        if suffix == ".svg":
            _pairwise_svg(views, output_path, cell_size=pairwise_cell_size)
        else:
            _pairwise_png(views, output_path, cell_size=pairwise_cell_size)
        return

    raise ValueError(f"Unsupported capture kind '{kind}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize attention_indices.pt or pairwise_initializer.pt as labeled SVG or PNG files."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to attention_indices.pt, pairwise_initializer.pt, or a raw [B, L, k] tensor",
    )
    parser.add_argument("output", type=Path, help="Output .svg or .png path")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index to visualize for attention captures")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Block size for attention occupancy view")
    parser.add_argument("--local-radius", type=int, default=32, help="Distance from the diagonal treated as local attention")
    parser.add_argument(
        "--attention-cell-size",
        type=int,
        default=DEFAULT_ATTENTION_CELL_SIZE,
        help="Pixel size per matrix cell for attention captures",
    )
    parser.add_argument(
        "--pairwise-cell-size",
        type=int,
        default=DEFAULT_PAIRWISE_CELL_SIZE,
        help="Pixel size per matrix cell for pairwise initializer captures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kind, payload = _load_payload(args.input)
    render(
        kind,
        payload,
        args.output,
        batch_index=args.batch_index,
        block_size=args.block_size,
        local_radius=args.local_radius,
        attention_cell_size=args.attention_cell_size,
        pairwise_cell_size=args.pairwise_cell_size,
    )


if __name__ == "__main__":
    main()