"""Project uploaded fence shapes into one fixed local plane for motion planning."""
import math

from shapely.geometry import Point, Polygon

DEFAULT_GEOFENCE_MARGIN = 1.0

def geofence_region(fence, projection):
    """Conservatively intersect inclusions and subtract every exclusion.

    An inclusion is required to define a centre. Circle approximations shrink
    inclusions and enlarge exclusions, never enlarging the permitted region.
    """
    def xy(point):
        lat, lon = point.latitude, point.longitude
        if not (math.isfinite(lat) and math.isfinite(lon)
                and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Invalid geofence coordinates')
        p = projection.forward(lat, lon, 0.)
        return p.x, p.y  # ENU; this geometry never becomes a PX4 setpoint.

    shapes = []
    for polygon in fence.polygons:
        shape = Polygon([xy(p) for p in polygon.vertices])
        if not shape.is_valid or shape.area <= 0:
            raise ValueError('Invalid geofence polygon')
        shapes.append((polygon.inclusion, shape))
    for circle in fence.circles:
        if not math.isfinite(circle.radius_m) or circle.radius_m <= 0:
            raise ValueError('Invalid geofence circle')
        radius = circle.radius_m if circle.inclusion else circle.radius_m / math.cos(math.pi / 512)
        shapes.append((circle.inclusion, Point(xy(circle.center)).buffer(radius, resolution=128)))
    inclusions = [shape for inclusion, shape in shapes if inclusion]
    if not inclusions:
        raise ValueError('A non-empty inclusion geofence is required')
    region = inclusions[0]
    for shape in inclusions[1:]:
        region = region.intersection(shape)
    for inclusion, shape in shapes:
        if not inclusion:
            region = region.difference(shape)
    if region.is_empty or region.area <= 0:
        raise ValueError('Geofence has no permitted area')
    return region


def spiral_region(region, centre, margin=DEFAULT_GEOFENCE_MARGIN):
    """Inset the fence without reducing the requested search extent."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Spiral geofence margin must be finite and nonnegative')
    safe = region.buffer(-margin, resolution=128)
    if safe.is_empty or not safe.covers(centre):
        raise ValueError('Spiral start lies outside the geofence clearance')
    return safe


def centred_spiral(region, requested_radius=7., margin=DEFAULT_GEOFENCE_MARGIN):
    """Use the area centroid; reject excluded centres rather than choosing another."""
    centre = region.centroid
    if not math.isfinite(requested_radius) or requested_radius <= 0:
        raise ValueError('Spiral radius must be finite and positive')
    if not region.contains(centre):
        raise ValueError('Geofence centroid lies outside the permitted area')
    spiral_region(region, centre, margin)
    return centre, requested_radius
