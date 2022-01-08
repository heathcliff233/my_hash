# -*- coding: utf-8 -*-
#
# Copyright (c) 2011-2014 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# Origianl repo link: https://github.com/tulip-control/polytope/blob/main/polytope/polytope.py

import numpy as np

np.random.seed(0)


def givens_rotation_matrix(i, j, theta, N):
    """Return the Givens rotation matrix for an N-dimensional space."""
    R = np.identity(N)
    c = np.cos(theta)
    s = np.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def solve_rotation_ap(u, v):
    r"""Return the rotation matrix for the rotation in the plane defined by the
    vectors u and v across TWICE the angle between u and v.
    This algorithm uses the Aguilera-Perez Algorithm \cite{Aguilera}
    to generate the rotation matrix. The algorithm works basically as follows:
    Starting with the Nth component of u, rotate u towards the (N-1)th
    component until the Nth component is zero. Continue until u is parallel to
    the 0th basis vector. Next do the same with v until it only has none zero
    components in the first two dimensions. The result will be something like
    this:
    [[u0,  0, 0 ... 0],
     [v0, v1, 0 ... 0]]
    Now it is trivial to align u with v. Apply the inverse rotations to return
    to the original orientation.
    NOTE: The precision of this method is limited by sin, cos, and arctan
    functions.
    """
    # TODO: Assert vectors are non-zero and non-parallel aka exterior
    # product is non-zero
    N = u.size  # the number of dimensions
    uv = np.stack([u, v], axis=1)  # the plane of rotation
    M = np.identity(N)  # stores the rotations for rorienting reference frame
    # ensure u has positive basis0 component
    if uv[0, 0] < 0:
        M[0, 0] = -1
        M[1, 1] = -1
        uv = M.dot(uv)
    # align uv plane with the basis01 plane and u with basis0.
    for c in range(0, 2):
        for r in range(N - 1, c, -1):
            if uv[r, c] != 0:  # skip rotations when theta will be zero
                theta = np.arctan2(uv[r, c], uv[r - 1, c])
                Mk = givens_rotation_matrix(r, r - 1, theta, N)
                uv = Mk.dot(uv)
                M = Mk.dot(M)
    # rotate u onto v
    theta = 2 * np.arctan2(uv[1, 1], uv[0, 1])
    R = givens_rotation_matrix(0, 1, theta, N)
    # perform M rotations in reverse order
    M_inverse = M.T
    R = M_inverse.dot(R.dot(M))
    return R


dim = 21
base1 = np.zeros(dim)
base1[0] = 1
base2 = np.random.random(21) - 1 / 3
rot_mx = solve_rotation_ap(base1, base2)
