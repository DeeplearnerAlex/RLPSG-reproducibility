// Dykstra projection of one vector onto general CSR slab constraints.
// No warm-start state is reused: x starts at the current nominal, q at zero.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" EXPORT int project_command_slabs(
    std::int64_t variables, std::int64_t rows,
    const std::int64_t* indptr, const std::int64_t* indices,
    const double* values, const double* lower, const double* upper,
    const double* nominal, double* result,
    std::int64_t max_sweeps, double primal_tolerance, double change_tolerance,
    double gap_tolerance, std::int64_t check_every, double* stats) {
    const auto started = std::chrono::steady_clock::now();
    std::fill(stats, stats+10, 0.0);
    std::copy(nominal, nominal+variables, result);
    std::vector<double> norm_squared(rows, 0.0), correction(rows, 0.0);
    std::vector<double> previous(variables), dual_gradient(variables);
    auto finish = [&](int code) {
        stats[6] = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        return code;
    };
    for (std::int64_t row=0; row<rows; ++row) {
        for (std::int64_t k=indptr[row]; k<indptr[row+1]; ++k)
            norm_squared[row] += values[k]*values[k];
        if (norm_squared[row] == 0.0 && (lower[row]>0.0 || upper[row]<0.0)) {
            stats[1] = std::max(lower[row], -upper[row]);
            return finish(1);  // Empty zero-row slab.
        }
        if (!std::isfinite(norm_squared[row])) return finish(2);
    }
    if (rows == 0) {
        stats[5] = 1.0;
        return finish(0);
    }
    for (std::int64_t sweep=0; sweep<max_sweeps; ++sweep) {
        std::copy(result, result+variables, previous.begin());
        for (std::int64_t row=0; row<rows; ++row) {
            const double norm2 = norm_squared[row];
            if (norm2 == 0.0) continue;
            double dot = correction[row]*norm2;
            for (std::int64_t k=indptr[row]; k<indptr[row+1]; ++k)
                dot += values[k]*result[indices[k]];
            if (!std::isfinite(dot)) return finish(2);
            const double projected = std::max(lower[row], std::min(upper[row], dot));
            const double alpha = (projected-dot)/norm2;
            const double update = correction[row]+alpha;
            for (std::int64_t k=indptr[row]; k<indptr[row+1]; ++k)
                result[indices[k]] += update*values[k];
            correction[row] = -alpha;
        }
        stats[0] = static_cast<double>(sweep+1);
        if ((sweep+1)%check_every && sweep+1<max_sweeps) continue;

        double primal = 0.0, change = 0.0, objective = 0.0;
        double signed_gap = 0.0, absolute_complementarity = 0.0;
        std::fill(dual_gradient.begin(), dual_gradient.end(), 0.0);
        for (std::int64_t j=0; j<variables; ++j) {
            if (!std::isfinite(result[j])) return finish(2);
            change = std::max(change, std::abs(result[j]-previous[j]));
            const double difference = result[j]-nominal[j];
            objective += 0.5*difference*difference;
        }
        for (std::int64_t row=0; row<rows; ++row) {
            double dot = 0.0;
            for (std::int64_t k=indptr[row]; k<indptr[row+1]; ++k) {
                dot += values[k]*result[indices[k]];
                dual_gradient[indices[k]] += correction[row]*values[k];
            }
            primal = std::max(primal, std::max(lower[row]-dot, dot-upper[row]));
            if (correction[row] != 0.0) {
                const double active_bound = correction[row]>0.0 ? upper[row] : lower[row];
                if (!std::isfinite(active_bound)) return finish(2);
                const double complementarity = correction[row]*(active_bound-dot);
                signed_gap += complementarity;
                absolute_complementarity += std::abs(complementarity);
            }
        }
        double stationarity = 0.0, stationarity_squared = 0.0;
        for (std::int64_t j=0; j<variables; ++j) {
            const double residual = result[j]-nominal[j]+dual_gradient[j];
            stationarity = std::max(stationarity, std::abs(residual));
            stationarity_squared += residual*residual;
        }
        signed_gap += 0.5*stationarity_squared;
        stats[1] = primal;
        stats[2] = change;
        stats[3] = signed_gap;
        stats[4] = objective;
        stats[7] = stationarity;
        stats[8] = absolute_complementarity;
        stats[9] = absolute_complementarity/(1.0+objective);
        const bool converged = primal<=primal_tolerance && change<=change_tolerance
            && stationarity<=change_tolerance*(1.0+std::sqrt(2.0*objective))
            && stats[9]<=gap_tolerance;
        if (converged) {
            stats[5] = 1.0;
            return finish(0);
        }
    }
    return finish(0);  // Finite iterate returned honestly with converged=false.
}

