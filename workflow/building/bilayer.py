import mbuild as mb
import numpy as np
from random import seed, shuffle, random
from mbuild import clone
import copy as cp

class Bilayer(mb.Compound):
    """Create a lipid bilayer and add solvent above and below.


    Arguments
    ----------
    lipids : list
        List of tuples in format (lipid, frac) where frac is the fraction of
        that lipid in the bilayer (lipid is a Compound)
    ref_atoms : int
        Indices of the atom in lipids to form the interface, one for each lipid
        in lipids (i.e., this atom is shifted to the 'interface' level)
    n_lipids_x : int
        Number of lipids in the x-direction per layer.
    n_lipids_y : int
        Number of lipids in the y-direction per layer.
    area_per_lipid : float
        Area per lipid.
    solvent : Compound
        Compound to solvate the bilayer with. Typically, a pre-equilibrated box
        of solvent.
    lipid_box : Box, optional
        A Box containing the lipids where no solvent will be added.
    spacing_z : float, optional
        Amount of space to add between opposing monolayers.
    solvent_per_lipid : int, optional, default=None
        Number of solvent molecules per lipid
    n_solvent : int, optional, default=None
        *Total* number of solvent molecules.
    tilt: float, optional, default=None
        Tilt angle relative to the bilayer normal to prescribe to lipids. Units
        are radians.
    random_seed : int, optional, default=12345
        Seed for random number generator for filling in lipids.
    mirror : bool, optional, default=True
        Make top and bottom layers mirrors of each other.
    """

    def __init__(self, lipids, ref_atoms, n_lipids_x=10, n_lipids_y=10,
                 area_per_lipid=1.0, lipid_box=None, spacing_z=None,
                 solvent=None, solvent_per_lipid=None, n_solvent=None,
                 solvent_density=None, solvent_mass=None, tilt=0,
                 random_seed=12345, mirror=True):

        super(Bilayer, self).__init__()

        # Santitize inputs
        i = 0
        while i < len(lipids):
            if lipids[i][1] == 0:
                lipids.remove(i)
            else:
                i += 1

        if sum([lipid[1] for lipid in lipids]) != 1.0:
            raise ValueError('Lipid fractions do not add up to 1.')
        assert len(ref_atoms) == len(lipids)

        self.lipids = lipids
        self.ref_atoms = ref_atoms
        self._lipid_box = lipid_box

        # 2D Lipid locations
        self.n_lipids_x = n_lipids_x
        self.n_lipids_y = n_lipids_y
        self.apl = area_per_lipid
        self.n_lipids_per_layer = self.n_lipids_x * self.n_lipids_y
        self.pattern = mb.Grid2DPattern(n_lipids_x, n_lipids_y)
        self.pattern.scale(np.sqrt(self.apl * self.n_lipids_per_layer))

        # Z-orientation parameters
        self.tilt = tilt
        if spacing_z:
            self.spacing = spacing_z
        else:
            self.spacing = max([lipid[0].boundingbox.lengths[2]
                for lipid in self.lipids]) * np.cos(self.tilt)

        # Solvent parameters
        if solvent:
            self.solvent = solvent
            if solvent_per_lipid != None:
                self.n_solvent = (self.n_lipids_x *
                                  self.n_lipids_y *
                                  2 * solvent_per_lipid)
            elif n_solvent != None:
                self.n_solvent = n_solvent
            else:
                raise ValueError("Requires either n_solvent_per_lipid ",
                                    "or n_solvent arguments")

        # Other parameters
        self.random_seed = random_seed
        self.mirror = mirror

        # Initialize variables and containers for lipids and solvent
        self.lipid_components = mb.Compound()
        self.solvent_components = mb.Compound()
        self._number_of_each_lipid_per_layer = []

        # Assemble the lipid layers
        seed(self.random_seed)
        self.tilt_about = [random(), random(), 0]
        top_layer, top_lipid_labels = self.create_layer()
        self.lipid_components.add(top_layer)
        if self.mirror == True:
            bottom_layer, bottom_lipid_labels = self.create_layer(
                    lipid_indices=top_lipid_labels,
                    flip_orientation=True)
        else:
            bottom_layer, bottom_lipid_labels = self.create_layer(
                                                    flip_orientation=True)
        bottom_layer.translate([0, 0, 0])
        self.lipid_components.add(bottom_layer)
        self.lipid_components.translate_to([0, 0, 0])

        # Assemble solvent components
        solvent_top = self.solvate(solvent_mass, solvent_density)
        solvent_bot = mb.clone(solvent_top)
        solvent_top.translate_to(
            [0, 0, self.spacing + 0.5 * solvent_top.boundingbox.lengths[2]]
            )
        solvent_bot.translate_to(
            [0, 0, -self.spacing - 0.5 * solvent_top.boundingbox.lengths[2]]
            )
        self.solvent_components.add(solvent_top)
        self.solvent_components.add(solvent_bot)

        # Add all compounds
        self.add(self.lipid_components)
        self.add(self.solvent_components)

    def solvate(self, solvent_mass, solvent_density):
        """Creates a single box of solvent molecules.
        The box size is based on the mass, density, and lipid area.
        """
        # Obtain a solvent volume (in cubic nm)
        avogadros_number = 6.02e23
        cm_to_nm = 1e7
        solvent_volume = (self.n_solvent *
                solvent_mass /
                avogadros_number /
                solvent_density *
                (cm_to_nm ** 3))
        lipid_area = self.n_lipids_x * self.n_lipids_y * self.apl

        # Determine height of the solvent box
        solvent_z = solvent_volume / lipid_area

        # Create Box object
        solvent_box = mb.Box(mins=[0, 0, 0],
                             maxs=[self.n_lipids_x * np.sqrt(self.apl),
                                   self.n_lipids_y * np.sqrt(self.apl),
                                   solvent_z * 0.5])

        # Fill box with solvent
        solvent_compound = mb.fill_box(compound=self.solvent,
                               n_compounds=int(self.n_solvent * 0.5),
                               box=solvent_box)

        return solvent_compound


    def create_layer(self, lipid_indices=None, flip_orientation=False):
        """Create a monolayer of lipids.
        Parameters
        ----------
        lipid_indices : list, optional, default=None
            A list of indices associated with each lipid in the layer.
        flip_orientation : bool, optional, default=False
            Flip the orientation of the layer with respect to the z-dimension.
        """

        layer = mb.Compound()
        if not lipid_indices:
            lipid_indices = list(range(self.n_lipids_per_layer))
            shuffle(lipid_indices)

        for n_type, n_of_lipid_type in enumerate(self.number_of_each_lipid_per_layer):
            current_type = self.lipids[n_type][0]
            for n_this_type in range(n_of_lipid_type):
                lipids_placed = sum(self.number_of_each_lipid_per_layer[:n_type]) + n_this_type
                new_lipid = clone(current_type)
                random_index = lipid_indices[lipids_placed]
                position = self.pattern[random_index]

                # Zero and space in z-direction and tilt
                new_lipid.rotate(theta=120.0 * int(3.0 * random()) * np.pi / 180.0,
                        around=[0, 0, 1])
                if flip_orientation == True:
                    new_lipid.rotate(theta=-self.tilt, around=self.tilt_about)
                else:
                    new_lipid.rotate(theta=self.tilt, around=self.tilt_about)
                particles = list(new_lipid.particles())
                ref_atom = self.ref_atoms[n_type]
                new_lipid.translate(-particles[ref_atom].pos + self.spacing)

                # Move to point on pattern
                if flip_orientation == True:
                    center = new_lipid.center
                    center[2] = 0.0
                    new_lipid.translate(-center)
                    new_lipid.rotate(np.pi, [0, 1, 0])
                    new_lipid.translate(center)
                new_lipid.translate(position)
                layer.add(new_lipid)
        return layer, lipid_indices


    @property
    def number_of_each_lipid_per_layer(self):
        """The number of each lipid per layer. """
        if self._number_of_each_lipid_per_layer:
            return self._number_of_each_lipid_per_layer

        for lipid in self.lipids[:-1]:
            self._number_of_each_lipid_per_layer.append(
                int(round(lipid[1] *self.n_lipids_per_layer))
                )

        # Rounding errors may make this off by 1, so just do total - whats_been_added.
        self._number_of_each_lipid_per_layer.append(self.n_lipids_per_layer -
                sum(self._number_of_each_lipid_per_layer))
        assert len(self._number_of_each_lipid_per_layer) == len(self.lipids)
        return self._number_of_each_lipid_per_layer


    @property
    def lipid_box(self):
        """The box containing all of the lipids. """
        if self._lipid_box:
            return self._lipid_box
        else:
            self._lipid_box = self.boundingbox

            # Add buffer around lipid box.
            self._lipid_box.mins -= np.array([0.5*np.sqrt(self.apl),
                                              0.5*np.sqrt(self.apl),
                                              0.5*np.sqrt(self.apl)])
            self._lipid_box.maxs += np.array([0.5*np.sqrt(self.apl),
                                              0.5*np.sqrt(self.apl),
                                              0.5*np.sqrt(self.apl)])
            return self._lipid_box
