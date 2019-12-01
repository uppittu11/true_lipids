import mbuild as mb
import numpy as np
from random import seed, shuffle, random
from mbuild import clone
import copy as cp
import bilayer

def load_molecule(filename):
    lipid = mb.load(filename)
    lipid.translate_to([0, 0, 0])
    return lipid


# load lipid prototypes
lipids = []

for molecule_name in ["cer", "chol", "ffa"]:
    filename = f"./molecules/{molecule_name}.mol2"
    molecule = load_molecule(filename)
    lipids.append(molecule)

# Define parameters
tilt_angle = 10 # degrees
spacing_z = 2.9
water_density = 1.0
water_mass = 18.01
n_lipids_per_edge = 6
area_per_lipid = .40
waters_per_lipid = 40

"""
Energy minimize
names = []
for particle in lipid1.particles():
    names += [particle.name]
    particle.name = particle.name[0]
lipid1.energy_minimize()
for i, particle in enumerate(lipid1.particles()):
    particle.name = names[i]
"""

# make lipids
bilayer1 = Bilayer(lipids=[(lipid1, 0.333), (lipid2, 0.334), (lipid3, 0.333)], ref_atoms=[77, 26, 2],
                  n_lipids_x=18, n_lipids_y=19, area_per_lipid=apl,
                  spacing_z=spacing_z, tilt=tilt, random_seed=23456, mirror=True)
bilayer1.remove(bilayer1[-1])
bilayer1.translate_to([0, 0, 0])
bilayer2 = mb.clone(bilayer1)
bilayer3 = mb.clone(bilayer1)
bilayer2.translate([0, 0, 6.0])
bilayer3.translate([0, 0, -6.0])

bilayer = mb.Compound()
bilayer.add(bilayer1)
bilayer.add(bilayer2)
bilayer.add(bilayer3)

lipid_box = bilayer.boundingbox

# make solvent boxes
n_solvent = int(6*(18*19)*wpl / 2)
solvent_z = solvent_volume / (bilayer.boundingbox.lengths[0] * bilayer.boundingbox.lengths[1])/10.0
solvent_box1 = mb.Box(mins=[lipid_box.mins[0], lipid_box.mins[1], lipid_box.maxs[2]+0.1],
                      maxs=[lipid_box.maxs[0], lipid_box.maxs[1], lipid_box.maxs[2] + solvent_z+0.1])
solvent_box2 = mb.Box(mins=[lipid_box.mins[0], lipid_box.mins[1], lipid_box.mins[2] - solvent_z-0.1],
                      maxs=[lipid_box.maxs[0], lipid_box.maxs[1], lipid_box.mins[2]-0.1])
solvent1 = mb.fill_box(compound=solvent, n_compounds=n_solvent, box=solvent_box1)
#solvent1 = mb.Compound()
#solvent1.add(mb.clone(solvent))
#solvent.translate([0, 0, 0.25*solvent_z])
#solvent1.add(mb.clone(solvent))
#solvent.translate([0, 0, 0.25*solvent_z])
#solvent1.add(mb.clone(solvent))
#solvent.translate([0, 0, 0.25*solvent_z])
#solvent1.add(mb.clone(solvent))

#solvent2 = mb.fill_box(compound=solvent, n_compounds=n_solvent, box=solvent_box2)
solvent2 = mb.clone(solvent1)
solvent2.translate([0, 0, -lipid_box.lengths[2]-solvent_z-0.2])

# add all components
system = mb.Compound()
system.add(bilayer)
system.add(solvent1)
system.add(solvent2)

# center
system.translate_to(system.boundingbox.lengths/2)

print("saving")
# save system
system.save('bilayer.hoomdxml', box=system.boundingbox, overwrite=True, ref_distance=10)


