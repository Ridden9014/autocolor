Proof-of-concept auto-sectioning for the Treetop Druid Haven dice tower top.
Run order: load.py -> an.py -> seg1.py (edge pairs + smoothing) -> seeds.py -> grow2.py -> export3mf.py
render.py is a quick preview renderer. Seed rules in grow2.py are specific to this model (post/rail coordinates).
Method: rule-based seed faces per part, then crease-aware region growing (multi-source Dijkstra where
crossing concave creases is expensive), then small-island cleanup, then 3MF export with per-triangle paint_color.
