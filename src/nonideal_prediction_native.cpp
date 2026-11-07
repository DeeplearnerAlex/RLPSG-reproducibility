// Optional exact-arithmetic state-only predictor. This is NOT the world model.
// Build explicitly with -fno-fast-math -ffp-contract=off. Never enable FMA or
// reassociation: the operation order mirrors the NumPy predictor substeps.
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
inline double clip_factor(double x, double y, double limit) {
    const double squared_x = x * x;
    const double squared_y = y * y;
    const double norm = std::sqrt(squared_x + squared_y);
    return std::min(1.0, limit / std::max(norm, 1e-30));
}
}

extern "C" int rlpsg_state_only_prediction_v1(
    std::int64_t population, std::int64_t steps, double h, double tau,
    double speed_cap, double acceleration_cap, double* positions,
    double* velocities, const double* commands) {
    if (population < 1 || steps < 1 || !(h > 0.0) || !(tau > 0.0)
        || !(speed_cap > 0.0) || !(acceleration_cap > 0.0)
        || !std::isfinite(h) || !std::isfinite(tau)
        || !std::isfinite(speed_cap) || !std::isfinite(acceleration_cap)
        || !positions || !velocities || !commands) {
        return 1;
    }
    const double half_h_squared = (0.5 * h) * h;
    for (std::int64_t row = 0; row < population; ++row) {
        const auto offset = 2 * row;
        const double command_scale = clip_factor(commands[offset], commands[offset + 1], speed_cap);
        const double command_x = commands[offset] * command_scale;
        const double command_y = commands[offset + 1] * command_scale;
        double px = positions[offset], py = positions[offset + 1];
        double vx = velocities[offset], vy = velocities[offset + 1];
        for (std::int64_t step = 0; step < steps; ++step) {
            double ax = (command_x - vx) / tau;
            double ay = (command_y - vy) / tau;
            const double acceleration_scale = clip_factor(ax, ay, acceleration_cap);
            ax = ax * acceleration_scale;
            ay = ay * acceleration_scale;
            double next_vx = vx + h * ax;
            double next_vy = vy + h * ay;
            const double velocity_scale = clip_factor(next_vx, next_vy, speed_cap);
            next_vx = next_vx * velocity_scale;
            next_vy = next_vy * velocity_scale;
            ax = (next_vx - vx) / h;
            ay = (next_vy - vy) / h;
            px = (px + h * vx) + half_h_squared * ax;
            py = (py + h * vy) + half_h_squared * ay;
            vx = next_vx;
            vy = next_vy;
        }
        positions[offset] = px;
        positions[offset + 1] = py;
        velocities[offset] = vx;
        velocities[offset + 1] = vy;
    }
    return 0;
}

