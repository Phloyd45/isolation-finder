#!/usr/bin/env python3
"""Build Isolation Finder map tiles from an OpenStreetMap extract.

Usage:
    build_tiles.py FILTERED.osm.pbf COVERAGE.poly OUTDIR

FILTERED.osm.pbf should already be cut down with osmium tags-filter to buildings,
landuse=residential and place nodes (see .github/workflows/publish.yml).
COVERAGE.poly is the extract's boundary in Osmosis .poly format (Geofabrik publishes one).

Writes:
    OUTDIR/index.json         which tiles exist and which the extract covers
    OUTDIR/t/<i>_<j>.json     one file per tile that has anything in it

Tiles are 0.1 degrees of latitude by 0.16 degrees of longitude (about 11 km square in
Britain), numbered i = floor(lat / 0.1), j = floor(lon / 0.16), the same grid the page uses.
Coordinates are integers in units of 1e-5 degrees (about 1 m), delta-encoded starting from
the tile's south-west corner, which keeps the files small.

Tile file keys:
    h  homes: buildings tagged as houses, farms and similar, plus isolated dwellings
    y  buildings tagged only building=yes
    p  settlements: [lat, lon, "village", "Name"] with absolute integer coordinates
    a  residential areas: [id, [x, y, dx, dy, ...]] outer ring, lat first
Buildings inside a residential area are dropped, because the area already covers them.
"""
import datetime
import json
import math
import os
import sys
import time
from array import array

import numpy as np
import osmium
import shapely

TLAT, TLON = 0.1, 0.16
Q = 100000  # 1e-5 degree units
HOME = {'house', 'residential', 'detached', 'semidetached_house', 'terrace', 'bungalow',
        'apartments', 'farm', 'cabin', 'static_caravan', 'dormitory'}
PLACES = {'city', 'town', 'village', 'hamlet'}
SIMPLIFY_DEG = 0.00008  # about 9 m north-south, 5 m east-west in Britain


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def kind_of(building):
    if building in HOME:
        return 0
    if building == 'yes':
        return 1
    return None


class Collect(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.lon = array('d')
        self.lat = array('d')
        self.kind = array('b')
        self.places = []          # (lat, lon, type, name)
        self.res_rings = []       # flat [lon, lat, lon, lat, ...] per ring
        self.areas = 0
        self.bad = 0

    def _point(self, lon, lat, k):
        self.lon.append(lon)
        self.lat.append(lat)
        self.kind.append(k)

    def node(self, n):
        t = n.tags
        if not n.location.valid():
            return
        b = t.get('building')
        if b:
            k = kind_of(b)
            if k is not None:
                self._point(n.location.lon, n.location.lat, k)
        p = t.get('place')
        if p in PLACES:
            self.places.append((n.location.lat, n.location.lon, p, t.get('name', '')))
        elif p == 'isolated_dwelling' and not b:
            self._point(n.location.lon, n.location.lat, 0)

    def area(self, a):
        self.areas += 1
        if self.areas % 1000000 == 0:
            log(f'{self.areas:,} areas read')
        t = a.tags
        b = t.get('building')
        k = kind_of(b) if b else None
        res = t.get('landuse') == 'residential'
        if k is None and not res:
            return
        try:
            for ring in a.outer_rings():
                if k is not None:
                    sx = sy = 0.0
                    n = 0
                    for nr in ring:
                        sx += nr.lon
                        sy += nr.lat
                        n += 1
                    if n:
                        self._point(sx / n, sy / n, k)
                    k = None  # one point per building, from its first outer ring
                if res:
                    flat = []
                    for nr in ring:
                        flat.append(nr.lon)
                        flat.append(nr.lat)
                    if len(flat) >= 8:
                        self.res_rings.append(flat)
        except osmium.InvalidLocationError:
            self.bad += 1


def read_poly(path):
    """Osmosis .poly file -> shapely geometry."""
    parts, holes, cur, hole = [], [], None, False
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines[1:]:
        if cur is None:
            if line == 'END':
                break
            hole = line.startswith('!')
            cur = []
        elif line == 'END':
            (holes if hole else parts).append(shapely.Polygon(cur))
            cur = None
        else:
            x, y = line.split()[:2]
            cur.append((float(x), float(y)))
    geom = shapely.union_all(parts)
    if holes:
        geom = geom.difference(shapely.union_all(holes))
    return geom


def delta(qlat, qlon, oi, oj):
    """Interleave lat/lon integer arrays as deltas from the tile corner."""
    out = np.empty(qlat.size * 2, dtype=np.int64)
    out[0::2] = np.diff(qlat, prepend=oi)
    out[1::2] = np.diff(qlon, prepend=oj)
    return out.tolist()


def main():
    pbf, polyfile, outdir = sys.argv[1:4]
    t0 = time.time()

    log('reading', pbf)
    h = Collect()
    h.apply_file(pbf, locations=True, idx='flex_mem')
    lon = np.frombuffer(h.lon, dtype=np.float64)
    lat = np.frombuffer(h.lat, dtype=np.float64)
    kind = np.frombuffer(h.kind, dtype=np.int8)
    log(f'{lon.size:,} buildings, {len(h.res_rings):,} residential rings, '
        f'{len(h.places):,} settlements, {h.bad:,} areas with missing nodes')

    # residential areas as shapely polygons
    sizes = np.fromiter((len(r) // 2 for r in h.res_rings), dtype=np.int64, count=len(h.res_rings))
    coords = np.fromiter((v for r in h.res_rings for v in r), dtype=np.float64,
                         count=int(sizes.sum()) * 2).reshape(-1, 2)
    ring_index = np.repeat(np.arange(sizes.size), sizes)
    rings = shapely.linearrings(coords, indices=ring_index) if sizes.size else np.array([], dtype=object)
    polys = shapely.polygons(rings)
    del h.res_rings, coords, ring_index, rings
    bad = ~shapely.is_valid(polys)
    if bad.any():
        polys[bad] = shapely.make_valid(polys[bad])
    log(f'{int(bad.sum()):,} residential areas repaired')

    # drop buildings that sit inside a residential area
    inside = np.zeros(lon.size, dtype=bool)
    if polys.size:
        tree = shapely.STRtree(polys)
        step = 1_000_000
        for s in range(0, lon.size, step):
            pts = shapely.points(lon[s:s + step], lat[s:s + step])
            hit = tree.query(pts, predicate='within')
            inside[s + hit[0]] = True
            log(f'checked {min(s + step, lon.size):,} buildings')
    keep = ~inside
    lon, lat, kind = lon[keep], lat[keep], kind[keep]
    log(f'{int(inside.sum()):,} buildings inside residential areas dropped, {lon.size:,} kept')

    tiles = {}

    def tile(i, j):
        key = f'{i}_{j}'
        t = tiles.get(key)
        if t is None:
            t = tiles[key] = {'i': i, 'j': j, 'h': None, 'y': None, 'p': [], 'a': []}
        return t

    # buildings, grouped by tile and sorted north-south so deltas stay small
    ti = np.floor(lat / TLAT).astype(np.int64)
    tj = np.floor(lon / TLON).astype(np.int64)
    qlat = np.rint(lat * Q).astype(np.int64)
    qlon = np.rint(lon * Q).astype(np.int64)
    order = np.lexsort((qlon, qlat, kind, tj, ti))
    ti, tj, qlat, qlon, kind = ti[order], tj[order], qlat[order], qlon[order], kind[order]
    if ti.size:
        brk = np.flatnonzero((np.diff(ti) != 0) | (np.diff(tj) != 0) | (np.diff(kind) != 0)) + 1
        for s, e in zip(np.r_[0, brk], np.r_[brk, ti.size]):
            i, j, k = int(ti[s]), int(tj[s]), int(kind[s])
            tile(i, j)['hy'[k]] = delta(qlat[s:e], qlon[s:e], i * 10000, j * 16000)

    # settlements
    for la, lo, typ, name in h.places:
        t = tile(math.floor(la / TLAT), math.floor(lo / TLON))
        t['p'].append([int(round(la * Q)), int(round(lo * Q)), typ, name])

    # residential outlines, simplified, stored in every tile their bounds touch
    if polys.size:
        simp = shapely.simplify(polys, SIMPLIFY_DEG, preserve_topology=True)
        parts = shapely.get_parts(simp)
        n_out = 0
        for g in parts:
            if shapely.get_type_id(g) != 3 or g.is_empty:  # polygons only
                continue
            ext = np.asarray(g.exterior.coords)
            if ext.shape[0] < 4:
                continue
            ext = ext[:-1]  # drop the closing point
            glat = np.rint(ext[:, 1] * Q).astype(np.int64)
            glon = np.rint(ext[:, 0] * Q).astype(np.int64)
            i0, i1 = math.floor(ext[:, 1].min() / TLAT), math.floor(ext[:, 1].max() / TLAT)
            j0, j1 = math.floor(ext[:, 0].min() / TLON), math.floor(ext[:, 0].max() / TLON)
            n_out += 1  # each outline gets its own id, repeated in every tile it touches
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    tile(i, j)['a'].append([n_out, delta(glat, glon, i * 10000, j * 16000)])
        log(f'{n_out:,} residential outlines written')

    # which tiles the extract covers (inside it, an absent tile means empty, not unknown)
    cover = read_poly(polyfile)
    shapely.prepare(cover)
    minx, miny, maxx, maxy = cover.bounds
    covered = []
    for i in range(math.floor(miny / TLAT), math.floor(maxy / TLAT) + 1):
        for j in range(math.floor(minx / TLON), math.floor(maxx / TLON) + 1):
            if cover.intersects(shapely.box(j * TLON, i * TLAT, (j + 1) * TLON, (i + 1) * TLAT)):
                covered.append(f'{i}_{j}')
    covset = set(covered)

    os.makedirs(os.path.join(outdir, 't'), exist_ok=True)
    files, total = [], 0
    for key, t in tiles.items():
        if key not in covset:
            continue
        body = {k: t[k] for k in ('h', 'y') if t[k]}
        if t['p']:
            body['p'] = t['p']
        if t['a']:
            body['a'] = t['a']
        if not body:
            continue
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        with open(os.path.join(outdir, 't', key + '.json'), 'w', encoding='utf-8') as f:
            f.write(text)
        files.append(key)
        total += len(text)
    index = {
        'v': 1,
        'built': datetime.date.today().isoformat(),
        'source': 'OpenStreetMap contributors (ODbL), extract from Geofabrik',
        'tlat': TLAT, 'tlon': TLON,
        'covered': covered,
        'files': sorted(files),
    }
    with open(os.path.join(outdir, 'index.json'), 'w') as f:
        json.dump(index, f, separators=(',', ':'))
    log(f'{len(files):,} tile files, {total / 1e6:.1f} MB, {len(covered):,} tiles covered, '
        f'{time.time() - t0:.0f} s in total')


if __name__ == '__main__':
    main()
