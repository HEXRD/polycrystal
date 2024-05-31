User Documentation
==================

.. toctree::
   :maxdepth: 2

The `polycrystal` library provides material models for polycrystalline
materials. It's companion package is  `polycrystalx`, which delivers
finite element simulation capabilities using material models implemented here.


There are currently five subpackages. Material models for linear elasticity and
slip modeling are in the `elasticity` and `slip` packages. Polycrystal grain
and phase assignment are handled in the microstructure package.  The
`orientations` package gives crystal symmetry operations and provides internal
support for orientation conversions.  Finally, the `utils` package
provides tensor data tools.

Here are some examples of how these might be used.

In the first example, we have an elastic material. To set the coefficients, you
use the `SingleCrystal` class.

.. code-block:: python

   from microstructure.elasticity.single_crystal import SingleCrystal

   crystal = SingleCrystal("cubic", (1.0, 0.5, 0.25))
   crystal = SingleCrystal.from_K_G(1/3, 1/2)

For slip modeling, here is what you do. First you select the slip system group
to use with the ``slip_groups.get_group()`` function. The we select the
slip model. In this case, we use a kind of Armstrong-Frederick model. Then
you instantiate the crystal with the slip systems and slip parameters.

.. code-block:: python

   from microstructure.slip.slip_groups import get_group
   from microstructure.slip import slipcrystal
   from microstructure.slip.slip_models import (
       ArmstrongFrederickParameters, ArmstrongFrederick,
   )

   fcc_slipsystems = get_group('fcc')
   inconel_718_af_params = ArmstrongFrederickParameters(
      gamma_dot_0=0.004, m=1/35, H=7124, H_d=19.5, A=32702.8, A_d=397.8, q12=1.2
   )
   inconel_718_af = slipcrystal.SlipCrystal(
      [slip_fcc()], ArmstrongFrederick(inconel_718_af_params)
   )

For any polycrystal modeling, you will need what we call the *polycrystal
configuration*, which is the spatial assignment of phases, grains and
orientations. We also refer to this as simply the *microstructure*.
The `microstructure` package provides several ways to generate a
polycrystal microstructure.

One simple way to build a microstructure is to use a Voronoi tesselation. The
code below constructs Voronoi microstructure with 100 grain inside the
unit cube, saving the microstructure in the file named ``microstructure.npz``,
so that it can be reproduced later, if needed.

.. code-block::

   from microstructure.microstructure.voronoi import Voronoi

   ngrains = 100
   box = [
       [0, 1],
       [0, 1],
       [0, 1]
   ]
   microstructure = Voronoi.random_voronoi(
       ngrains, box, "microstructure.npz"
   )

Other types of microstructure are single crystal, analytic, voxel data, and
a multiphase microstructure, which is some number of single phase
microstructures with a function for marking phase by spatial location.
