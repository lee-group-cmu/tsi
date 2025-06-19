import numpy as np
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import os 

os.environ["GAMMAPY_DATA"] = "/home/export/ajshen/ada-sbi-cosmic-rays/gammapy-datasets/1.2"

_channel = "V->e"
_massDM = 1e5*u.Unit("GeV")
_jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
_modelDM = DarkMatterAnnihilationSpectralModel(mass=_massDM, channel=_channel, jfactor=_jfactor)

def differential_dm_flux(log10_energy_gev):
    return np.array(_modelDM.evaluate(10**log10_energy_gev * u.Unit("GeV"), 1e9))


# flux functions return log (counts per unit area unit time)
def log10_crab_flux(log10_energy_gev):
    raise NotImplementedError
    C = -0.12
    log_f0 = -10.248 
    alpha = 2.5
    E_IC = 48
    
    base = log10_energy_gev - np.log10(E_IC)
    
    return log_f0 + C * (np.abs(base))**alpha + 3 - log10_energy_gev

# differential flux is correct
def differential_crab_flux(log10_energy_gev):
    C = -0.12
    log_f0 = -10.248 
    alpha = 2.5
    E_IC = 48
    
    base = log10_energy_gev - np.log10(E_IC)
    
    return 10**(log_f0 + C * (np.abs(base))**alpha + 3 - (2*log10_energy_gev))

def log10_mrk421_flux(log10_energy_gev):
    raise NotImplementedError
    energy_gev = 10**log10_energy_gev
    
    amplitdue = 2.69e-13 # ref in 1/(Gev cm^2 s)
    E_0 = 443 # reference in GeV
    alpha = 2.381
    beta = 0.17
    E_ratio = energy_gev/E_0
    exponent = -alpha - beta * np.log(E_ratio)
    
    dNdE = amplitdue * E_ratio**exponent
    
    return np.log10(dNdE * energy_gev)

def differential_mrk421_flux(log10_energy_gev):
    energy_gev = 10**log10_energy_gev
    
    amplitdue = 2.69e-13 # original in 1/(Tev cm^2 s) -> 1/(Gev cm^2 s)
    E_0 = 443 # original in TeV -> GeV
    alpha = 2.381
    beta = 0.17
    E_ratio = energy_gev/E_0
    exponent = -alpha - beta * np.log10(E_ratio)
    
    dNdE = amplitdue * E_ratio**exponent
    
    return dNdE

def get_source_trajectory(
    source_name: str,
    observer_latitude: float
):
    observer = EarthLocation(lat=observer_latitude*u.deg, lon=97.3*u.deg, height=4100*u.m)
    midnight = Time('2024-01-01 00:00:00')
    delta_midnight = np.linspace(0, 24, 10000)*u.hour
    observer_frame = AltAz(obstime=midnight+delta_midnight, location=observer)
    trajectory = SkyCoord.from_name(source_name).transform_to(observer_frame)
    return trajectory
