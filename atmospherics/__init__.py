"""
atmospherics
============
Atmospheric propagation effects for FSO link simulation.

Usage
-----
    from atmospherics import kim_fog, turbulence, scintillation, rain_attenuation, snow_attenuation, kruse_haze, beer_lambert, hufnagel_valley, pointing
"""

from atmospherics.fog import beer_lambert, kim_fog, kruse_haze
from atmospherics.precipitation import rain_attenuation, snow_attenuation
from atmospherics.turbulence import turbulence
from atmospherics.scintillation import scintillation
from atmospherics.hufnagel_valley import (
    hv_cn2,
    integrated_cn2,
    fried_parameter,
    slant_turbulence,
)
from atmospherics.pointing import (
    pointing_loss_static,
    pointing_loss_jitter,
    pointing_loss_db_static,
    pointing_loss_db_jitter,
)

__all__ = [
    "beer_lambert",
    "kim_fog",
    "kruse_haze",
    "rain_attenuation",
    "snow_attenuation",
    "turbulence",
    "scintillation",
    "hv_cn2",
    "integrated_cn2",
    "fried_parameter",
    "slant_turbulence",
    "pointing_loss_static",
    "pointing_loss_jitter",
    "pointing_loss_db_static",
    "pointing_loss_db_jitter",
]
