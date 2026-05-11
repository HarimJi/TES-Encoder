# TES-Encoder

CUDA-accelerated tetrahedral encoded SDF (TES) encoder.

Given an inner tetrahedral mesh — and optionally a matching outer mesh —
the encoder builds the TES geometry and writes a `.tes` file holding the
nodes, faces, tetrahedra, and per-tetrahedron encoded face lists.

## Requirements

- Windows 10/11
- Visual Studio 2022 (MSVC 19.36+)
- CUDA Toolkit 13.2 (or any version supplying the CCCL `cuda::minimum`
  functor — older toolkits still work if you swap it back for
  `thrust::minimum`)
- CMake 3.18+
- A CUDA-capable GPU; the default target architecture is `sm_100` (Blackwell).
  Override at configure time with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`
  (e.g. `75` for Turing, `86` for Ampere).

## Build

From the repository root:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

The executable lands at `build\Release\tes_encoder.exe`.

To target a different GPU architecture, pass `-DCMAKE_CUDA_ARCHITECTURES`
on the configure step:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=75
```

## Preparing the meshes

Generate the tetrahedral meshes with [Gmsh](https://gmsh.info/) and export
as ASCII MSH v2.2. The encoder is strict about the mesh, so verify the
following before running it:

- **File format.** Gmsh ASCII MSH v2.2 (`File > Export > .msh > Version 2 ASCII`).
  Binary MSH and v4.x layouts are not supported.
- **Two blocks of nodes.** The inner mesh must contain a dim-2 block for
  surface nodes followed by a dim-3 block for internal nodes. Node indices
  must be contiguous and ascending. Gmsh produces this naturally when the
  geometry has a defined surface and volume.
- **Two blocks of elements.** A dim-2 block of triangular faces (the
  surface) followed by a dim-3 block of tetrahedra (the volume). Elements
  of any other type (lines, hexahedra, etc.) are not handled.
- **Closed, watertight surface.** Every surface edge must be shared by
  exactly two faces. The loader rejects meshes with boundary edges.
- **Manifold vertices.** Every surface vertex must be touched by at least
  three faces. Non-manifold pinches are rejected.
- **Outward-pointing face normals.** Surface triangles must be wound
  counter-clockwise as seen from outside the volume (right-hand rule
  pointing out). Inverted winding flips the sign of the encoded SDF.
- **Matching inner/outer boundary nodes.** When an outer mesh is supplied,
  any node that lies on the shared interface must coincide with the
  corresponding inner-mesh node within `1e-6` (see `is_near` in
  `geometry_utils.cuh`). Mesh the inner and outer regions in the same Gmsh
  session, or copy the inner surface mesh into the outer model, so that
  shared nodes are bit-identical.
- **No degenerate triangles.** Run `MSH::Inspect` (logged at start of
  `Make_Geometry`) and check the minimum triangle area is well above
  floating-point noise; sub-`1e-12` areas will destabilize the projection.

## Run

```
tes_encoder.exe --input <inner.msh> --output <out.tes>
                [--outer <outer.msh>]
                [--grid-quality <int>     (default 128)]
                [--alpha <double>         (default 0.999)]
                [--epsilon <double>       (default 1e-6)]
                [--deformable]
```

| Flag              | Required | Default | Meaning                                                  |
| ----------------- | -------- | ------- | -------------------------------------------------------- |
| `--input`, `-i`   | yes      | —       | Inner tetrahedral mesh (Gmsh ASCII v2.2 `.msh`)          |
| `--output`, `-o`  | yes      | —       | Output `.tes` path (written verbatim, suffix included)   |
| `--outer`         | no       | none    | Outer tetrahedral mesh; common boundary nodes are merged |
| `--grid-quality`  | no       | `128`   | Barycentric sampling quality per tetrahedron             |
| `--alpha`         | no       | `0.999` | Tetra shrink factor toward its centroid before sampling  |
| `--epsilon`       | no       | `1e-6`  | Slack for keeping near-winning faces in the encoding     |
| `--deformable`    | no       | off     | Reorders encoded faces to front-load surface neighbors   |
| `--help`, `-h`    | —        | —       | Print usage and exit                                     |

### Pipeline

1. `Load_Inner_MSH` — read the inner `.msh`, build vertex/edge/face maps
2. `Load_Outer_MSH` *(if `--outer`)* — same, for the outer mesh
3. `Make_Geometry` — merge nodes, classify tetras as boundary vs. internal
4. `Encode_With_Optimization` — Lipschitz prune + barycentric importance
   sampling on the GPU; sort faces by importance
5. `Print_Statistics` — memory accounting
6. `Export` — write the `.tes` file

### Examples

Inner-only encode:

```powershell
.\build\Release\tes_encoder.exe `
    --input .\example\MSH\M16_Nut_Tol_100_Inner.msh `
    --output .\example\TES\M16_Nut_Tol_100.tes `
    --grid-quality 128 `
    --alpha 0.999 `
    --epsilon 1e-6
```

Inner + outer encode (paired mesh):

```powershell
.\build\Release\tes_encoder.exe `
    --input .\example\MSH\M16_Bolt_Inner.msh `
    --outer .\example\MSH\M16_Bolt_Outer.msh `
    --output .\example\TES\M16_Bolt.tes `
    --grid-quality 128 `
    --alpha 0.999 `
    --epsilon 1e-6
```

Deformable encoding (front-loads boundary-adjacent faces):

```powershell
.\build\Release\tes_encoder.exe `
    --input .\example\MSH\M16_Nut_Tol_100_Inner.msh `
    --output .\example\TES\M16_Nut_Tol_100_deformable.tes `
    --deformable
```

## Project layout

```
include/
  vector4d.cuh           Padded 16-byte aligned 3D vector
  itypes.cuh             iedge_t / iface_t / itetra_t and helpers
  geometry_utils.cuh     dot, is_near, project_point_to_triangle
  barycentric_utils.cuh  Barycentric grid indexing
  cuda_error.cuh         CUDA error-check macros
  msh.hpp                Gmsh ASCII v2.2 loader + topology maps
  tes.cuh                TES encoder class
src/
  main.cu                CLI entry point
  msh.cpp                MSH loader implementation
  tes.cu                 TES encoder kernels and host driver
example/
  MSH/                   Sample input meshes
  TES/                   Encoded outputs
  OBJ/                   Surface OBJ exports
misc/
  code_style.md          Project coding style
```
