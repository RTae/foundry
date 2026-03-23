from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import torch


CELL_SIZE = 4
BLOCK_SIZE = 16
PADDING = 24
LABEL_WIDTH = 180

BACKGROUND = (255, 253, 248)
PANEL = (244, 241, 232)
BORDER = (200, 195, 182)
ACCENT = (11, 110, 79)


def _load_indices(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "indices" not in payload:
            raise ValueError(f"{path} does not contain an 'indices' entry")
        indices = payload["indices"]
    else:
        indices = payload
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"{path} did not resolve to a tensor of indices")
    if indices.ndim != 3:
        raise ValueError(
            f"Expected indices with shape [B, L, k], got {tuple(indices.shape)}"
        )
    return indices.to(dtype=torch.long, device="cpu")


def _adjacency_from_indices(indices: torch.Tensor, batch_index: int) -> torch.Tensor:
    if batch_index < 0 or batch_index >= indices.shape[0]:
        raise IndexError(
            f"batch_index={batch_index} out of range for batch size {indices.shape[0]}"
        )
    batch = indices[batch_index]
    n_queries = batch.shape[0]
    adjacency = torch.zeros((n_queries, n_queries), dtype=torch.bool)
    rows = torch.arange(n_queries).unsqueeze(1).expand_as(batch)
    valid = (batch >= 0) & (batch < n_queries)
    adjacency[rows[valid], batch[valid]] = True
    return adjacency


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


def _draw_matrix(title: str, matrix: torch.Tensor, origin_x: int, origin_y: int) -> list[str]:
    n_rows, n_cols = matrix.shape
    width = n_cols * CELL_SIZE
    height = n_rows * CELL_SIZE
    parts = [_svg_text(origin_x, origin_y - 8, title, size=15, weight="600")]
    parts.append(_svg_rect(origin_x, origin_y, width, height, "#f4f1e8", "#c8c3b6"))
    true_points = torch.nonzero(matrix, as_tuple=False)
    for row, col in true_points.tolist():
        x = origin_x + col * CELL_SIZE
        y = origin_y + row * CELL_SIZE
        parts.append(
            _svg_rect(x, y, CELL_SIZE, CELL_SIZE, "#0b6e4f", "#0b6e4f")
        )
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
    for row in range(y, y + height):
        base = (row * image_width + x) * 3
        for _ in range(width):
            image[base] = color[0]
            image[base + 1] = color[1]
            image[base + 2] = color[2]
            base += 3


def _draw_matrix_png(
    image: bytearray,
    image_width: int,
    matrix: torch.Tensor,
    origin_x: int,
    origin_y: int,
) -> None:
    n_rows, n_cols = matrix.shape
    width = n_cols * CELL_SIZE
    height = n_rows * CELL_SIZE
    _image_rect(image, image_width, origin_x, origin_y, width, height, PANEL)
    _image_rect(image, image_width, origin_x, origin_y, width, 1, BORDER)
    _image_rect(image, image_width, origin_x, origin_y + height - 1, width, 1, BORDER)
    _image_rect(image, image_width, origin_x, origin_y, 1, height, BORDER)
    _image_rect(image, image_width, origin_x + width - 1, origin_y, 1, height, BORDER)
    for row, col in torch.nonzero(matrix, as_tuple=False).tolist():
        x = origin_x + col * CELL_SIZE
        y = origin_y + row * CELL_SIZE
        _image_rect(image, image_width, x, y, CELL_SIZE, CELL_SIZE, ACCENT)


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


def _draw_summary(
    adjacency: torch.Tensor,
    occupancy: torch.Tensor,
    origin_x: int,
    origin_y: int,
    block_size: int,
) -> list[str]:
    parts = [_svg_text(origin_x, origin_y, "Packed layout summary", size=15, weight="600")]
    offsets = _row_offsets(adjacency)
    density = adjacency.float().mean().item()
    active_blocks = int(occupancy.sum().item())
    total_blocks = occupancy.numel()
    lines = [
        f"rows={adjacency.shape[0]} cols={adjacency.shape[1]}",
        f"nnz={int(adjacency.sum().item())}",
        f"density={density:.6f}",
        f"block_size={block_size}",
        f"active_blocks={active_blocks}/{total_blocks}",
        f"row_offsets_head={offsets[: min(8, len(offsets))]}",
    ]
    current_y = origin_y + 24
    for line in lines:
        parts.append(_svg_text(origin_x, current_y, line, size=13))
        current_y += 20
    return parts


def render_svg(indices: torch.Tensor, output_path: Path, batch_index: int, block_size: int) -> None:
    adjacency = _adjacency_from_indices(indices, batch_index=batch_index)
    occupancy = _block_occupancy(adjacency, block_size=block_size)

    matrix_width = adjacency.shape[1] * CELL_SIZE
    matrix_height = adjacency.shape[0] * CELL_SIZE
    occupancy_width = occupancy.shape[1] * CELL_SIZE
    occupancy_height = occupancy.shape[0] * CELL_SIZE

    left_x = PADDING
    top_y = 56
    right_x = left_x + max(matrix_width, occupancy_width) + LABEL_WIDTH
    summary_x = right_x + PADDING
    width = summary_x + 360
    height = max(top_y + matrix_height, top_y + matrix_height + PADDING + occupancy_height, 260) + PADDING

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, "#fffdf8", "#fffdf8"),
        _svg_text(PADDING, 28, "RFD3 Sparse Layout View", size=18, weight="700"),
    ]
    parts.extend(_draw_matrix("Attention adjacency", adjacency, left_x, top_y))
    parts.extend(
        _draw_matrix(
            "Block occupancy",
            occupancy,
            left_x,
            top_y + matrix_height + PADDING,
        )
    )
    parts.extend(_draw_summary(adjacency, occupancy, summary_x, top_y, block_size))
    parts.append("</svg>")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_png(indices: torch.Tensor, output_path: Path, batch_index: int, block_size: int) -> None:
    adjacency = _adjacency_from_indices(indices, batch_index=batch_index)
    occupancy = _block_occupancy(adjacency, block_size=block_size)

    matrix_width = adjacency.shape[1] * CELL_SIZE
    matrix_height = adjacency.shape[0] * CELL_SIZE
    occupancy_width = occupancy.shape[1] * CELL_SIZE
    occupancy_height = occupancy.shape[0] * CELL_SIZE

    left_x = PADDING
    top_y = 56
    width = left_x + max(matrix_width, occupancy_width) + PADDING
    height = top_y + matrix_height + PADDING + occupancy_height + PADDING

    image = bytearray(BACKGROUND * (width * height))
    _draw_matrix_png(image, width, adjacency, left_x, top_y)
    _draw_matrix_png(image, width, occupancy, left_x, top_y + matrix_height + PADDING)
    _write_png(output_path, image, width, height)


def render(indices: torch.Tensor, output_path: Path, batch_index: int, block_size: int) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".svg":
        render_svg(indices, output_path, batch_index=batch_index, block_size=block_size)
        return
    if suffix == ".png":
        render_png(indices, output_path, batch_index=batch_index, block_size=block_size)
        return
    raise ValueError(f"Unsupported output format '{output_path.suffix}'. Use .svg or .png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize sparse attention indices as adjacency and block occupancy SVG or PNG files."
    )
    parser.add_argument("input", type=Path, help="Path to attention_indices.pt or a raw [B, L, k] tensor")
    parser.add_argument("output", type=Path, help="Output .svg or .png path")
    parser.add_argument("--batch-index", type=int, default=0, help="Batch index to visualize")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Block size for occupancy view")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indices = _load_indices(args.input)
    render(indices, args.output, batch_index=args.batch_index, block_size=args.block_size)


if __name__ == "__main__":
    main()