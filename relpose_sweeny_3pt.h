//
// Created by yjunj on 10/18/21.
//
#include <vector>
#include <Eigen/Dense>


struct CameraPose {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};


typedef std::vector<CameraPose> CameraPoseVector;


int relpose_sweeny_3pt(const Eigen::Vector3d &axis,
                       const std::vector<Eigen::Vector3d> &x1,
                       const std::vector<Eigen::Vector3d> &x2,
                       CameraPoseVector *output);


// Solves the QEP by solving det(lambda^2*A + lambda*B + C) where we know that (1+lambda^2) is a factor.
// This is the case in the upright solvers from Sweeney et al.
// The roots are found using the closed form solver for the quartic.
int qep_div_1_q2(const Eigen::Matrix<double, 3, 3> &A,
                 const Eigen::Matrix<double, 3, 3> &B,
                 const Eigen::Matrix<double, 3, 3> &C,
                 double eig_vals[4],
                 Eigen::Matrix<double, 3, 4> *eig_vecs);


/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0. Only returns real roots */
int solve_quartic_real(double b, double c, double d, double e, double roots[4]);


/* Finds a single real root of x^3 + b*x^2 + c*x + d = 0 */
void solve_cubic_single_real(double b, double c, double d, double &root);


// Computes polynomial p(x) = det(x^2*I + x * A + B)
void detpoly3(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B, double coeffs[7]);

/* Sign of component with largest magnitude */
inline double sign(const double z);

void essential_from_motion(const CameraPose &pose, Eigen::Matrix3d *E);
