"""Evolve plastic slip over a multi-step stress sequence, saving material state after each step.

Loads an initial material state and a :class:`StressSequence` from disk,
integrates the slip ODEs step-by-step using :class:`~polycrystal.slip.solve.Pointwise`,
and writes ``step-NN.npz`` files to the output directory.
"""
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np

from ..slip.solve import StateData, StressIncrement, StressSequence, Pointwise
from ..slip.slipcrystal import SlipCrystal


def resolve_material(material):
    """Resolve a material model from an instance, callable, or module string.

    Parameters
    ----------
    material: SlipCrystal, callable, or str
        Material object or import path (e.g. ``'my_materials:lshr_af'`` or
        ``'my_materials.lshr_af'``).

    Returns
    -------
    SlipCrystal
    """
    if isinstance(material, str):
        if ":" in material:
            mod_name, attr_name = material.split(":", 1)
            try:
                mod = importlib.import_module(mod_name)
                matl = getattr(mod, attr_name)
            except (ImportError, AttributeError) as exc:
                raise ValueError(
                    f"Could not import material '{attr_name}' from module '{mod_name}': {exc}"
                ) from exc
        elif "." in material:
            mod_name, attr_name = material.rsplit(".", 1)
            try:
                mod = importlib.import_module(mod_name)
                matl = getattr(mod, attr_name)
            except (ImportError, AttributeError) as exc:
                raise ValueError(
                    f"Could not import material '{attr_name}' from module '{mod_name}': {exc}"
                ) from exc
        else:
            raise ValueError(
                f"Material '{material}' must be specified as 'module:attribute' "
                "or 'module.attribute', or passed as a SlipCrystal instance."
            )
        if callable(matl) and not isinstance(matl, SlipCrystal):
            matl = matl()
        return matl
    if callable(material) and not isinstance(material, SlipCrystal):
        return material()
    return material


def evolve_slip(material, material_data, stress_data, outdir=None):
    """Integrate a multi-step stress sequence and save step-NN.npz output files.

    Parameters
    ----------
    material: SlipCrystal, callable, or str
        Material model or import specifier.
    material_data: StateData, str, or Path
        Initial material state or path to ``.npz`` file containing initial state.
    stress_data: StressSequence, str, or Path
        Stress sequence object or path to ``.npz`` file containing sequence.
    outdir: str or Path, optional
        Output directory for ``step-NN.npz`` files. Defaults to the directory
        containing ``stress_data`` if a path was provided, else the current directory.

    Returns
    -------
    StateData
        Final material state after completing all steps.
    """
    matl = resolve_material(material)

    if isinstance(material_data, (str, Path)):
        mdata = StateData.from_file(matl, material_data)
    elif isinstance(material_data, StateData):
        mdata = material_data
    else:
        raise TypeError(f"Unsupported material_data type: {type(material_data)}")

    if isinstance(stress_data, (str, Path)):
        stress_path = Path(stress_data)
        sseq = StressSequence.from_file(stress_path)
        if outdir is None:
            outdir = stress_path.parent
    elif isinstance(stress_data, StressSequence):
        sseq = stress_data
    else:
        raise TypeError(f"Unsupported stress_data type: {type(stress_data)}")

    if outdir is None:
        outdir = Path.cwd()
    else:
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if len(sseq.csig) != len(sseq.times):
        raise RuntimeError(
            "Number of stress states must match number of times in StressSequence"
        )

    nsteps = len(sseq.times) - 1
    s_shp = mdata.state.shape
    npts = s_shp[0]

    for istep in range(nsteps):
        dtime = sseq.times[istep + 1] - sseq.times[istep]
        sdata = StressIncrement(sseq.csig[istep], sseq.csig[istep + 1], dtime)
        solver = Pointwise(mdata, sdata)
        thist = solver.run(time_history=True)
        fp = solver.integrate_fp(thist)
        state = np.zeros(s_shp)
        for ipt in range(npts):
            state[ipt] = thist[ipt][1:, -1]
        mdata = StateData(matl, mdata.orient, state, fp)
        # Save new material state
        step_file = outdir / f"step-{istep + 1:02}.npz"
        mdata.save_data(step_file)

    return mdata


def main(args):
    """Load material and stress-sequence files, integrate each step, and save results.

    Parameters
    ----------
    args:
        Parsed argument namespace with attributes ``material``,
        ``material_data``, ``stress_data``, and optional ``output_dir``.
    """
    outdir = getattr(args, "output_dir", None)
    print(
        f"material: {args.material}\n"
        f"material data file: {args.material_data}\n"
        f"stress data file: {args.stress_data}"
    )
    evolve_slip(
        material=args.material,
        material_data=args.material_data,
        stress_data=args.stress_data,
        outdir=outdir,
    )


def argparser(*args):
    """Build and return the argument parser for the evolve-slip CLI."""
    p = argparse.ArgumentParser(
        description="Evolve plastic slip over a sequence of stress states"
    )
    p.add_argument(
        "material",
        help="Material model specifier (e.g. 'materials_module:lshr_af')",
    )
    p.add_argument(
        "material_data",
        help="Path to .npz file containing initial material data",
    )
    p.add_argument(
        "stress_data",
        help="Path to .npz file containing sequence of stress states",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to write step-NN.npz files (defaults to stress_data directory)",
    )
    return p


def cli():
    """Command-line entry point."""
    # Add working directory to system path so we use local files to load materials.
    sys.path.append(str(Path.cwd()))
    print("cwd: ", str(Path.cwd()))
    p = argparser(*sys.argv[1:])
    args = p.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
