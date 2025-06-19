from enum import Enum
from dataclasses import dataclass
import pandas as pd

class PrimaryParticleId(Enum):
    GAMMA = 1
    HADRON = 14

class SecondaryParticleType(Enum):
    PHOTON = 0
    MESON = 1
    BARYON = 2
    LEPTON = 3
    UNKNOWN = 4
  
PARTICLE_DF = pd.DataFrame([
    (1, "gamma", 0, SecondaryParticleType.PHOTON),
    (2, "positron", 1, SecondaryParticleType.LEPTON),
    (3, "electron", -1, SecondaryParticleType.LEPTON),
    (5, "muon+", 1, SecondaryParticleType.LEPTON),
    (6, "muon-", -1, SecondaryParticleType.LEPTON),
    (7, "pion", 0, SecondaryParticleType.MESON),
    (8, "pion+", 1, SecondaryParticleType.MESON),
    (9, "pion-", -1, SecondaryParticleType.MESON),
    (10, "kaon_long", 0, SecondaryParticleType.MESON),
    (11, "kaon+", 1, SecondaryParticleType.MESON),
    (12, "kaon-", -1, SecondaryParticleType.MESON),
    (13, "neutron", 0, SecondaryParticleType.BARYON),
    (14, "proton", 1, SecondaryParticleType.BARYON),
    (15, "antiproton", -1, SecondaryParticleType.BARYON),
    (16, "kaon_short", 0, SecondaryParticleType.MESON),
    (17, "eta?", 0, SecondaryParticleType.UNKNOWN),
    (18, "lambda", 0, SecondaryParticleType.BARYON),
    (19, "sigma+", 1, SecondaryParticleType.BARYON),
    (20, "sigma", 0, SecondaryParticleType.BARYON),
    (21, "sigma-", -1, SecondaryParticleType.BARYON),
    (22, "xi", 0, SecondaryParticleType.BARYON),
    (23, "xi-", -1, SecondaryParticleType.BARYON),
    (24, "omega-", -1, SecondaryParticleType.BARYON),
    (25, "antineutron", 0, SecondaryParticleType.BARYON),
    (26, "antilambda", 0, SecondaryParticleType.BARYON),
    (27, "antisigma-", -1, SecondaryParticleType.BARYON),
    (28, "antisigma", 0, SecondaryParticleType.BARYON),
    (29, "antisigma+", 1, SecondaryParticleType.BARYON),
    (30, "antixi", 0, SecondaryParticleType.BARYON),
    (31, "antixi+", 1, SecondaryParticleType.BARYON),
    (32, "antiomega+", 1, SecondaryParticleType.BARYON),
    (50, "omega_meson", 0, SecondaryParticleType.MESON),
    (51, "rho", 0, SecondaryParticleType.MESON),
    (52, "rho+", 1, SecondaryParticleType.MESON),
    (53, "rho-", -1, SecondaryParticleType.MESON),
    (54, "delta++", 2, SecondaryParticleType.BARYON),
    (55, "delta+", 1, SecondaryParticleType.BARYON),
    (56, "delta", 0, SecondaryParticleType.BARYON),
    (57, "delta-", -1, SecondaryParticleType.BARYON),
    (58, "antidelta--", -2, SecondaryParticleType.BARYON),
    (59, "antidelta-", -1, SecondaryParticleType.BARYON),
    (60, "antidelta", 0, SecondaryParticleType.BARYON),
    (61, "antidelta+", 1, SecondaryParticleType.BARYON),
], columns=["particle_id", "name", "charge", "particle_type"])

def get_particle_name_by_id(id: int):
    if id not in PARTICLE_DF['particle_id']:
        return "unknown"
    return PARTICLE_DF[PARTICLE_DF['particle_id'] == id]['name']