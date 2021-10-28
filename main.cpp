#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "relpose_sweeny_3pt.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;


int synthesisCorresponds(int num, Eigen::AngleAxisd &real_R, Eigen::Vector3d &real_t, Matrix3d &cam_mtx,
                         double jitter_degree, double height, double noise_max,
                         vector<Eigen::Vector3d> &x1_set, vector<Eigen::Vector3d> &x2_set) {

    uniform_int_distribution<int> xdist(-20, 20);
    uniform_int_distribution<int> ydist(-15, 15);
    uniform_int_distribution<int> zdist(50 - height, 50);
    normal_distribution<double> n_noise(0, noise_max + 1e-10);
    default_random_engine rng;

    jitter_degree = jitter_degree / 180 * M_PI;
    Matrix3d jitter_mtx(AngleAxisd(jitter_degree, Vector3d::UnitX()));
    for (int i = 0; i < num; i++) {
        double x = xdist(rng) + double(rand()) / RAND_MAX;
        double y = ydist(rng) + double(rand()) / RAND_MAX;
        double z = zdist(rng);

        Vector3d X1(x, y, z);
        Vector3d X2 = real_R * jitter_mtx * X1 + real_t;

        Vector3d x1 = cam_mtx * X1 / X1[2];
        Vector3d x2 = cam_mtx * X2 / X2[2];

        double nx = n_noise(rng);
        double ny = n_noise(rng);
        if (abs(nx) <= noise_max) {
            x2[0] += nx;
        }
        if (abs(ny) <= noise_max) {
            x2[1] += ny;
        }

        x1_set.push_back(x1);
        x2_set.push_back(x2);
    }
    return 0;
}

int fitPoseRansac(const Eigen::Vector3d &axis,
                  const std::vector<Eigen::Vector3d> &x1_set, const std::vector<Eigen::Vector3d> &x2_set,
                  CameraPose &output, int n_iter = 100) {
    int n = x1_set.size();
    cv::RNG rng;
    int n_most_inline = 0;
    CameraPose best_pose;
    Matrix3d cam_mtx;
    cam_mtx << 4523.2365, 0, 1920, 0, 4523.2365, 1080, 0, 0, 1;

    for (int i = 0; i < n_iter; i++) {
        CameraPoseVector cam_poses;
        Eigen::Matrix3d E;
        int i1 = 0, i2 = 0, i3 = 0;

        while (i1 == i2 || i2 == i3 || i3 == i1) {
            i1 = rng(n);
            i2 = rng(n);
            i3 = rng(n);
        }
        vector<Vector3d> x1_set_batch{x1_set[i1], x1_set[i2], x1_set[i3]};
        vector<Vector3d> x2_set_batch{x2_set[i1], x2_set[i2], x2_set[i3]};

        relpose_sweeny_3pt(axis, x1_set_batch, x2_set_batch, &cam_poses);
        for (auto &cam_pose: cam_poses) {
            int n_inline = 0;
            essential_from_motion(cam_pose, &E);
            AngleAxisd aa(cam_pose.R);
            AngleAxisd baa(best_pose.R);
            for (int j = 0; j < n; j++) {
                Matrix3d F = cam_mtx.inverse().transpose() * E * cam_mtx.inverse();
                double loss = x2_set[j].transpose() * F * x1_set[j];
                if (abs(loss) < 0.1) {
                    n_inline += 1;
                }
            }
            if (n_inline > n_most_inline) {
                n_most_inline = n_inline;
                best_pose = cam_pose;
            }
        }
    }
    AngleAxisd a(best_pose.R);
    output = best_pose;
    cout << "n_inlines: " << n_most_inline << endl;
    return 0;
}

int ransac_fit(Vector3d &axis, vector<Vector3d> &x1_set, vector<Vector3d> &x2_set, double rotation_angle,
               Vector3d &real_t, double &rotation_error, double &translation_error) {
    CameraPose best_pose;
    fitPoseRansac(axis, x1_set, x2_set, best_pose);
    AngleAxisd pose_R(best_pose.R);
    Vector3d pose_t = best_pose.t;
    rotation_error = pose_R.angle() / M_PI * 180 - rotation_angle;
    translation_error = acos(pose_t.dot(real_t.normalized()));

    return 0;
}


int random_rotation_test() {
    double max_height = 0;
    int n_corresponds = 500;
    Vector3d axis(0, 0, 1);

    Matrix3d cam_mtx;
    cam_mtx << 4523.2365, 0, 1920, 0, 4523.2365, 1080, 0, 0, 1;// 3666.666504, 0, 2736, 0, 3666.666504, 1824, 0, 0, 1;


    vector<double> axis_jitter_degrees{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
//    vector<double> axis_jitter_degrees{0, 0.5, 1};
    vector<Vector3d> x1_set;
    vector<Vector3d> x2_set;
    vector<double> mean_rotation_errors;
    vector<double> mean_translation_errors;
    int n_iter = 100;

    vector<double> random_angles;
    vector<Vector3d> random_translations;
    for (int i = 0; i < n_iter; i++) {
        double rotation_angle = double(rand()) / RAND_MAX * 180;;

        double x = (double(rand()) / RAND_MAX - 0.5) * 2 * 10;
        double y = (double(rand()) / RAND_MAX - 0.5) * 2 * 10;
        double z = (double(rand()) / RAND_MAX - 0.5) * 2 * 0.5;
        Vector3d translation(x, y, z);

        double rotation_error = 0;
        double translation_error = 0;
        AngleAxisd real_R(rotation_angle / 180 * M_PI, axis);
        x1_set.clear();
        x2_set.clear();
        synthesisCorresponds(n_corresponds, real_R, translation, cam_mtx, 0, max_height, 0,
                             x1_set, x2_set);
        ransac_fit(axis, x1_set, x2_set, rotation_angle, translation, rotation_error, translation_error);
        if (abs(rotation_error) < 1e-5) {
            cout << rotation_angle << " ";
            cout << x << "," << y << "," << z << endl;

            random_angles.push_back(rotation_angle);
            random_translations.push_back(translation);
        }

    }
    cout << endl;
    n_iter = random_angles.size();
    cout << "==== n_iter: " << n_iter << endl;
    for (auto &axis_jitter_degree: axis_jitter_degrees) {
        cout << "===== Axis Jitter Degree: " << axis_jitter_degree << endl;
        double mean_rotation_error = 0;
        double mean_translation_error = 0;
        for (int i = 0; i < n_iter; i++) {
            x1_set.clear();
            x2_set.clear();

//            Vector3d translation(-2.4424, 0.633259, 0.0711844);
//            double rotation_angle = 1.60405;
            double rotation_angle = random_angles[i];
            Vector3d translation = random_translations[i];

            AngleAxisd real_R(rotation_angle / 180 * M_PI, axis);
            double rotation_error = 0;
            double translation_error = 0;
            synthesisCorresponds(n_corresponds, real_R, translation, cam_mtx, axis_jitter_degree, max_height, 0,
                                 x1_set, x2_set);
            ransac_fit(axis, x1_set, x2_set, rotation_angle, translation, rotation_error, translation_error);
            cout << "Rotation Error: " << rotation_error << endl;
            cout << "Translation Error: " << translation_error << endl << endl;
            mean_rotation_error += abs(rotation_error);
            mean_translation_error += abs(translation_error);
        }
        cout << "==== Mean Rotation Error: " << mean_rotation_error / n_iter << endl;
        cout << "==== Mean Translation Error: " << mean_translation_error / n_iter << endl;

        mean_rotation_errors.push_back(mean_rotation_error / n_iter);
        mean_translation_errors.push_back(mean_translation_error / n_iter);
    }
    for (auto i: mean_rotation_errors)
        cout << i << ' ';
    cout << endl;
    for (auto i: mean_translation_errors)
        cout << i << ' ';
    return 0;

    return 0;
}


int random_noise_test() {
    double max_height = 0;
    int n_corresponds = 500;
    Vector3d axis(0, 0, 1);

    Matrix3d cam_mtx;
    cam_mtx << 4523.2365, 0, 1920, 0, 4523.2365, 1080, 0, 0, 1;// 3666.666504, 0, 2736, 0, 3666.666504, 1824, 0, 0, 1;

//    vector<double> noise_levels{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    vector<double> noise_levels{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<Vector3d> x1_set;
    vector<Vector3d> x2_set;
    vector<double> mean_rotation_errors;
    vector<double> mean_translation_errors;
    int n_iter = 4;

    for (auto &noise_max: noise_levels) {
        cout << "===== Noise Max: " << noise_max << endl;
        double mean_rotation_error = 0;
        double mean_translation_error = 0;
        for (int i = 0; i < n_iter; i++) {

//            double rotation_angle = random_angles[i];
//            Vector3d translation = random_translations[i];
            double rotation_angle = 28.2357;
            Vector3d translation = {6.41102, 2.01697, -0.49765};

            AngleAxisd real_R(rotation_angle / 180 * M_PI, axis);
            double rotation_error = 0;
            double translation_error = 0;

            x1_set.clear();
            x2_set.clear();
            synthesisCorresponds(n_corresponds, real_R, translation, cam_mtx, 0, max_height, noise_max,
                                 x1_set, x2_set);
            ransac_fit(axis, x1_set, x2_set, rotation_angle, translation, rotation_error, translation_error);
            cout << "Rotation Error: " << rotation_error << endl;
            cout << "Translation Error: " << translation_error << endl << endl;
            mean_rotation_error += abs(rotation_error);
            mean_translation_error += abs(translation_error);
        }
        cout << "==== Mean Rotation Error: " << mean_rotation_error / n_iter << endl;
        cout << "==== Mean Translation Error: " << mean_translation_error / n_iter << endl;

        mean_rotation_errors.push_back(mean_rotation_error / n_iter);
        mean_translation_errors.push_back(mean_translation_error / n_iter);
    }
    for (auto i: mean_rotation_errors)
        cout << i << ' ';
    cout << endl;
    for (auto i: mean_translation_errors)
        cout << i << ' ';
    return 0;
}

//int random_init() {
//    vector<double> random_angles;
//    vector<Vector3d> random_translations;
//    for (int i = 0; i < n_iter; i++) {
//        srand(time(NULL));
//        double rotation_angle = double(rand()) / RAND_MAX * 180;;
//
//        double x = (double(rand()) / RAND_MAX - 0.5) * 2 * 10;
//        double y = (double(rand()) / RAND_MAX - 0.5) * 2 * 10;
//        double z = (double(rand()) / RAND_MAX - 0.5) * 2 * 0.5;
//        Vector3d translation(x, y, z);
//
//        double rotation_error = 0;
//        double translation_error = 0;
//        AngleAxisd real_R(rotation_angle / 180 * M_PI, axis);
//        x1_set.clear();
//        x2_set.clear();
//        synthesisCorresponds(n_corresponds, real_R, translation, cam_mtx, 0, 15, 0,
//                             x1_set, x2_set);
//        ransac_fit(axis, x1_set, x2_set, rotation_angle, translation, rotation_error, translation_error);
//        if (abs(rotation_error) < 0.2) {
//            cout << rotation_angle << " ";
//            cout << x << "," << y << "," << z << endl;
//
//            random_angles.push_back(rotation_angle);
//            random_translations.push_back(translation);
//        }
//
//    }
//}

int main() {
    int n_corresponds = 500;
    Vector3d axis(0, 0, 1);

    Matrix3d cam_mtx;
    cam_mtx << 4523.2365, 0, 1920, 0, 4523.2365, 1080, 0, 0, 1;// 3666.666504, 0, 2736, 0, 3666.666504, 1824, 0, 0, 1;

    vector<double> max_heights{0, 5, 15, 25, 35, 45};
    vector<Vector3d> x1_set;
    vector<Vector3d> x2_set;
    vector<double> mean_rotation_errors;
    vector<double> mean_translation_errors;//    cam_mtx << 3666.666504, 0, 2736, 0, 3666.666504, 1824, 0, 0, 1;


    int n_iter = 10;

    for (auto &max_height: max_heights) {
        cout << "===== Max Height: " << max_height << " =====" << endl;
        double mean_rotation_error = 0;
        double mean_translation_error = 0;
        for (int i = 0; i < n_iter; i++) {

            double rotation_angle = 28.2357;
            Vector3d translation = {6.41102, 2.01697, -0.49765};

            AngleAxisd real_R(rotation_angle / 180 * M_PI, axis);
            double rotation_error = 0;
            double translation_error = 0;

            x1_set.clear();
            x2_set.clear();
            synthesisCorresponds(n_corresponds, real_R, translation,
                                 cam_mtx, 0, max_height, 0,
                                 x1_set, x2_set);

            ransac_fit(axis, x1_set, x2_set,
                       rotation_angle, translation, rotation_error, translation_error);

            cout << "Rotation Error: " << rotation_error << endl;
            cout << "Translation Error: " << translation_error << endl << endl;
            mean_rotation_error += abs(rotation_error);
            mean_translation_error += abs(translation_error);
        }
        cout << "==== Mean Rotation Error: " << mean_rotation_error / n_iter << endl;
        cout << "==== Mean Translation Error: " << mean_translation_error / n_iter << endl;

        mean_rotation_errors.push_back(mean_rotation_error / n_iter);
        mean_translation_errors.push_back(mean_translation_error / n_iter);
    }

}

