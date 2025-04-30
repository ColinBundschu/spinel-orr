import datetime
import functools
from typing import Callable

import colorama
import numpy as np
import pytz
import structlog
from structlog.stdlib import BoundLogger, EventDict, WrappedLogger

from constants import DFT_OUT_FOLDER, MAX_FORCE_SLAB_eVpA
from ecat_types import Status
from remote.database import UpdateResult


def initialize_logging(level: int) -> BoundLogger:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            eastern_time_stamper,
            remove_keys('hostname'),
            run_stat_processor,
            console_renderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
    return structlog.get_logger()


def eastern_time_stamper(_, __, event_dict: EventDict) -> EventDict:
    '''Custom processor to generate timestamps in Eastern Time with the desired format'''
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(pytz.utc).astimezone(eastern)
    event_dict['timestamp'] = now.strftime('%m/%d %H:%M:%S')
    return event_dict


def run_stat_processor(_, __, event_dict: EventDict) -> EventDict:
    '''Add/modify some fields in the event_dict for column formatting of calculations.'''
    Etot_eV = event_dict.pop('Etot_eV', None)
    db_Etot_eV = event_dict.pop('db_Etot_eV', None)
    # Don't print the same value twice if db/local are in sync
    if event_dict['event'] == UpdateResult.CalcInSync.value:
        event_dict['has_E_or_dE_meV'] = 'Synced'
    elif Etot_eV is not None and db_Etot_eV is not None:
        event_dict['has_E_or_dE_meV'] = (Etot_eV - db_Etot_eV) * 1000
    elif Etot_eV is not None:
        event_dict['has_E_or_dE_meV'] = 'Local Only'
    elif 'Status' in event_dict:
        event_dict['has_E_or_dE_meV'] = None

    run_stat_keys = ('Runtime_s', 'Status', 'Last_modified_utc_ts', 'Forces_max_L2_eVpA')
    db_events = {key: event_dict.pop('db_' + key) for key in run_stat_keys if 'db_' + key in event_dict}
    for key in run_stat_keys:
        local_color_val = 'local'
        if key == 'Forces_max_L2_eVpA' and 'path' in event_dict:
            # Force coloration requires knowing the geometry type (from the path)
            local_color_val = MAX_FORCE_SLAB_eVpA
        if key in event_dict:
            event_dict[key] = (event_dict[key], local_color_val)
        elif key in db_events:
            event_dict[key] = (db_events[key], 'db')
        elif 'Status' in event_dict:
            event_dict[key] = (None, None)

    if 'path' in event_dict:
        if event_dict['path'].startswith(DFT_OUT_FOLDER + '/'):
            path = event_dict['path'][len(DFT_OUT_FOLDER + '/'):]
        elif event_dict['path'].startswith('materials/'):
            path = event_dict['path'][len('materials/'):]
        else:
            raise ValueError(f'Unexpected path format: {event_dict["path"]}')
        if 'material' in event_dict:
            raise ValueError('material and path cannot both be specified in a log')
        event_dict['Material_Geo'] = path.split('/')[:2]

    if 'material' in event_dict:
        if 'Material_Geo' in event_dict:
            raise ValueError('material and Material_Geo cannot both be specified in a log')
        event_dict['Material_Geo'] = (event_dict['material'].basename, event_dict['material'].geo)

    return event_dict


def remove_keys_processor(keys: tuple[str, ...], _, __, event_dict: EventDict) -> EventDict:
    for key in keys:
        event_dict.pop(key, None)  # Remove the key if it exists, do nothing otherwise
    return event_dict


def remove_keys(*keys: str) -> Callable[[WrappedLogger, str, EventDict], EventDict]:
    return functools.partial(remove_keys_processor, keys)


def material_geo_formatter(_key: str, material_geo: tuple[str, str]) -> str:
    material, geo = material_geo
    color = colorama.Fore.MAGENTA if geo == 'bulk' else colorama.Fore.BLUE
    return f'{color}{material:<8}{colorama.Style.RESET_ALL}'


def level_formatter(_key: str, level: str) -> str:
    if level == 'info':
        return ''

    level_styles = structlog.dev.ConsoleRenderer.get_default_level_styles()
    return f'{level_styles[level]}{level:<5}{colorama.Style.RESET_ALL}'


def event_formatter(_key: str, event: str) -> str:
    width = 15 if '✅' in event or '💀' in event else 16
    if 'SBATCH' in event:
        color = colorama.Style.BRIGHT + colorama.Fore.RED
    elif 'already running' in event:
        color = colorama.Style.BRIGHT + colorama.Fore.CYAN
    elif 'already converged' in event:
        color = colorama.Fore.GREEN
    else:
        color = colorama.Fore.WHITE
    return f'{color}{event:<{width}}{colorama.Style.RESET_ALL}'


def last_modified_formatter(_key: str, last_modified_utc_ts_src: tuple[datetime.datetime, str]) -> str:
    last_modified_utc_ts, src = last_modified_utc_ts_src
    time_delta = datetime.datetime.now(datetime.timezone.utc) - last_modified_utc_ts
    if time_delta < datetime.timedelta(seconds=-2):
        raise ValueError(f'Last modified date is {time_delta} in the future')

    # unit, seconds per unit, render color
    units = [
        ('y', 60*60*24*365, colorama.Fore.BLUE),
        ('M', 60*60*24*30, colorama.Fore.LIGHTBLUE_EX),
        ('d', 60*60*24, colorama.Fore.MAGENTA),
        ('h', 60*60, colorama.Fore.RED),
        ('m', 60, '\x1b[38;5;214m'),  # Orange
        ('s', 1, colorama.Fore.YELLOW),
    ]

    # Find the largest unit that will return a value greater than 1
    dt_s = time_delta.total_seconds()
    for unit, seconds_per_unit, color in units:
        dt = int(dt_s / seconds_per_unit)
        if dt_s >= seconds_per_unit:
            break

    if src == 'db':
        color = colorama.Fore.LIGHTBLACK_EX

    return f'{color}{dt:2}{unit}{colorama.Style.RESET_ALL}'


def runtime_formatter(_key: str, runtime_s_src: tuple[int, str]) -> str:
    runtime_s, src = runtime_s_src
    runtime_str = f'{int(runtime_s/3600):3}h' if runtime_s is not None else '    '
    color = colorama.Fore.BLUE if src == 'local' else colorama.Fore.LIGHTBLACK_EX
    return f'{color}{runtime_str}{colorama.Style.RESET_ALL}'


def status_formatter(_key: str, status_src: tuple[str, str]) -> str:
    status, src = status_src
    color = colorama.Fore.CYAN if src == 'local' else colorama.Fore.LIGHTBLACK_EX
    return f'{colorama.Style.BRIGHT}{color}{Status[status].value}{colorama.Style.RESET_ALL}'


def energy_formatter(_key: str, energy: float | str) -> str:
    if energy == 'Local Only':
        return ' E: Local '
    if energy == 'Db Only':
        return ' E: Db    '
    if energy == 'Synced':
        return 'ΔE: ----- '
    if energy is None:
        return '          '
    color = colorama.Fore.RED if energy > 0 else colorama.Fore.GREEN
    dE_str = f'{int(energy):>4}' if abs(energy) > 1 else f'{energy:>4.1f}'
    return f'ΔE:{color}{dE_str}meV{colorama.Style.RESET_ALL}'


def forces_formatter(_key: str, force_thresh_eVpA: tuple[float | None, float | str]) -> str:
    max_force_eVpA, thresh_eVpA = force_thresh_eVpA
    if max_force_eVpA is None:
        return '    '
    if np.isnan(max_force_eVpA):
        return ' NaN'

    if thresh_eVpA == 'db':
        color = colorama.Fore.LIGHTBLACK_EX
    elif thresh_eVpA == 'local':
        color = colorama.Fore.WHITE
    elif max_force_eVpA < thresh_eVpA:
        color = colorama.Fore.LIGHTGREEN_EX
    elif max_force_eVpA < thresh_eVpA*2:
        color = '\x1b[38;5;148m' # Light Green Yellow
    elif max_force_eVpA < thresh_eVpA*4:
        color = colorama.Fore.YELLOW
    elif max_force_eVpA < thresh_eVpA*8:
        color = '\x1b[38;5;214m' # Macaroni Orange
    else:
        color = colorama.Fore.LIGHTRED_EX
    return f'{color}{int(max_force_eVpA*1000):4}{colorama.Style.RESET_ALL}'


def timestamp_formatter(_key: str, timestamp: str) -> str:
    return f'{colorama.Style.DIM}{colorama.Fore.LIGHTCYAN_EX}{timestamp}{colorama.Style.RESET_ALL}'


def path_formatter(_key: str, path: str) -> str:
    return f'\x1b[38;5;108m{path}{colorama.Style.RESET_ALL}'


def default_kvp_formatter(key: str, timestamp: str) -> str:
    kvp_formatter = structlog.dev.KeyValueColumnFormatter(key_style=colorama.Fore.CYAN, value_style=colorama.Fore.GREEN,
                                                          reset_style=colorama.Style.RESET_ALL, value_repr=str)
    return kvp_formatter(key, timestamp)


def console_renderer() -> structlog.dev.ConsoleRenderer:
    return structlog.dev.ConsoleRenderer(
        columns=[
            structlog.dev.Column('timestamp', timestamp_formatter),
            structlog.dev.Column('Material_Geo', material_geo_formatter),
            structlog.dev.Column('level', level_formatter),
            structlog.dev.Column('event', event_formatter),
            structlog.dev.Column('Runtime_s', runtime_formatter),
            structlog.dev.Column('Last_modified_utc_ts', last_modified_formatter),
            structlog.dev.Column('Forces_max_L2_eVpA', forces_formatter),
            structlog.dev.Column('Status', status_formatter),
            structlog.dev.Column('has_E_or_dE_meV', energy_formatter),
            structlog.dev.Column('path', path_formatter),
            structlog.dev.Column('', default_kvp_formatter)  # Default formatter for all keys not explicitly mentioned
        ]
    )
