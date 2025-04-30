from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class MinimizationPoint():
    Etot_eV: float
    time_s: float | None
    iter: int
    type: str
    forces_sum_L2_eVpA: float | None
    forces_max_L2_eVpA: float | None
    unit_cell_volume_A3: float | None
    dft_timestamp: int | None


def min_points_from_data(data: dict | None) -> list[MinimizationPoint] | None:
    if not data or 'min_Etots_eV' not in data:
        return

    default = [None] * len(data['min_Etots_eV'])
    zipped_min_data = zip(data['min_Etots_eV'],
                          data['min_times_s'],
                          data['min_iters'],
                          data['min_types'],
                          data.get('min_forces_sum_L2_eVpA', default),
                          data.get('min_forces_max_L2_eVpA', default),
                          data.get('min_unit_cell_volume_A3', default),
                          data.get('min_dft_utc_timestamps_s', default))
    return [MinimizationPoint(*x) for x in zipped_min_data]


def serialize_min_points(points: list[MinimizationPoint]) -> dict:
    if not points:
        return {}

    return {
        'min_Etots_eV': [point.Etot_eV for point in points],
        'min_times_s': [point.time_s for point in points],
        'min_iters': [point.iter for point in points],
        'min_types': ''.join([point.type[0] for point in points]),
        'min_forces_sum_L2_eVpA': [point.forces_sum_L2_eVpA for point in points],
        'min_forces_max_L2_eVpA': [point.forces_max_L2_eVpA for point in points],
        'min_unit_cell_volume_A3': [point.unit_cell_volume_A3 for point in points],
        'min_dft_utc_timestamps_s': [point.dft_timestamp for point in points],
    }
