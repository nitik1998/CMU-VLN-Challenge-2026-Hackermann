#!/usr/bin/env python3
"""Question-domain driven rectilinear inspection of a 360 panorama.

The equirectangular panorama is the canonical sensor record.  Vision models
receive overlapping tangent views selected from the compiled physical search
domain.  This avoids a target-name special case: wall art, floor objects and
room-height landmarks use the same representation and differ only in the
vertical band that must be enumerated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from PIL import Image

from rectilinear import rectilinear_view


@dataclass(frozen=True)
class InspectionView:
    domain_kind: str
    center_u: float
    center_v: float
    hfov_deg: float
    bearing_deg: float
    reason: str

    def metadata(self) -> dict:
        return asdict(self)


def inspection_bands(domain: dict) -> list[tuple[str, float, str]]:
    """Vertical panorama bands implied by explicit domain elements.

    The values are camera-image fractions, not semantic sectors.  They merely
    choose tangent projection centres; object identity and spatial relations
    still come from pixels plus registered geometry.
    """
    elements = domain.get("elements", [])
    surfaces = {str(element.get("surface")) for element in elements
                if element.get("kind") == "surface_family"}
    bands = []
    if "wall" in surfaces or domain.get("wall"):
        bands.append(("wall", 0.43, "enumerate vertical wall evidence"))
    if "floor" in surfaces or domain.get("floor"):
        bands.append(("floor", 0.68, "enumerate visible floor evidence"))
    # Furniture/relation-only questions still need an undistorted room-height
    # pass to ground their landmarks.  Registered support tops are inspected
    # separately at a geometry-derived centre and closer field of view.
    if not bands:
        bands.append(("room", 0.52, "ground target and relation landmarks"))
    return bands


def panorama_inspection_specs(domain: dict, panorama_size,
                              views_per_band: int = 6) -> list[InspectionView]:
    width, height = map(float, panorama_size)
    specs = []
    for kind, vertical_fraction, reason in inspection_bands(domain):
        for index in range(views_per_band):
            centre_u = index / views_per_band * width
            specs.append(InspectionView(
                domain_kind=kind,
                center_u=centre_u,
                center_v=vertical_fraction * height,
                hfov_deg=70.0,
                bearing_deg=(centre_u / width - 0.5) * 360.0,
                reason=reason,
            ))
    return specs


def render_panorama_inspection(panorama: Image.Image, domain: dict,
                               views_per_band: int = 6) -> list[tuple[dict, Image.Image]]:
    return [(spec.metadata(), rectilinear_view(
                panorama, spec.center_u, spec.center_v,
                hfov_deg=spec.hfov_deg, out_size=(896, 640)))
            for spec in panorama_inspection_specs(
                domain, panorama.size, views_per_band)]
