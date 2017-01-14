/* This Source Code Form is subject to the terms of the Mozilla Public License, 
 * v. 2.0. If a copy of the MPL was not distributed with this file, 
 * You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef HRBF_PHI_FUNCS_HPP_
#define HRBF_PHI_FUNCS_HPP_

/// @file hrbf_phi_funcs.h
/// Original author: Gael Guennebaud - gael.guennebaud@inria.fr - http://www.labri.fr/perso/guenneba/
/// Rodolphe Vaillant - (Fixed the gradient evaluation) - http://www.irit.fr/~Rodolphe.Vaillant

/** @brief Radial basis functions definitions (function phi)
  Here you can add more radial basis function definitions.  
*/

/**
 * @class Rbf_pow3
 * Radial basis function phi(x) = x^3 first and second derivative
 * 
 **/
template<typename Scalar>
struct Rbf_pow3
{
    // phi(x) = x^3
    static inline Scalar f  (const Scalar& x) { return x*x*x;             }
    // first derivative phi'(x) = 3x^2
    static inline Scalar df (const Scalar& x) { return Scalar(3) * x * x; }
    // second derivative phi''(x) = 6x
    static inline Scalar ddf(const Scalar& x) { return Scalar(6) * x;     }
};

#endif //HRBF_PHI_FUNCS_HPP_
