"""QR helpers for TradeApp."""

from __future__ import annotations

import io

import qrcode
from PIL import Image
from PyQt5.QtGui import QPixmap


def generate_qr(data: str, *, box_size: int = 8, border: int = 2) -> Image.Image:
    qr = qrcode.QRCode(box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    if not isinstance(img, Image.Image):
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def qr_to_pixmap(img: Image.Image) -> QPixmap:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buffer.getvalue(), "PNG")
    return pix


def generate_qr_pixmap(data: str) -> QPixmap:
    return qr_to_pixmap(generate_qr(data))


__all__ = ["generate_qr", "qr_to_pixmap", "generate_qr_pixmap"]
