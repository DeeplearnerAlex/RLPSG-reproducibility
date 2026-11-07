// Same ordered polygon clipping and shoelace centroid as the Python helper.
// Compile without fast-math/FMA contraction; Qhull and region order stay Python.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

struct Point { double x, y; };

static std::vector<Point> clip(const std::vector<Point>& input, double nx, double ny, double bound) {
    if (input.empty()) return input;
    std::vector<Point> output;
    output.reserve(input.size()+2);
    Point previous = input.back();
    double dp = previous.x*nx+previous.y*ny-bound;
    for (const Point& current: input) {
        const double dc = current.x*nx+current.y*ny-bound;
        if ((dp <= 1e-10) != (dc <= 1e-10)) {
            output.push_back({previous.x+((current.x-previous.x)*dp)/(dp-dc),
                              previous.y+((current.y-previous.y)*dp)/(dp-dc)});
        }
        if (dc <= 1e-10) output.push_back(current);
        previous = current;
        dp = dc;
    }
    return output;
}

extern "C" EXPORT int clipped_voronoi_centroids(
    std::int64_t cells, std::int64_t vertices,
    const double* coordinates, const std::int64_t* offsets, const std::int64_t* indices,
    double xmin, double xmax, double ymin, double ymax,
    double* output, std::int32_t* fallback, double* stats) {
    const auto started = std::chrono::steady_clock::now();
    stats[0] = 0.;
    for (std::int64_t cell=0; cell<cells; ++cell) {
        std::vector<Point> polygon;
        for (std::int64_t k=offsets[cell]; k<offsets[cell+1]; ++k) {
            const auto vertex = indices[k];
            if (vertex < 0 || vertex >= vertices) return 1;
            polygon.push_back({coordinates[2*vertex], coordinates[2*vertex+1]});
        }
        polygon = clip(polygon, 1., 0., xmax);
        polygon = clip(polygon, -1., 0., -xmin);
        polygon = clip(polygon, 0., 1., ymax);
        polygon = clip(polygon, 0., -1., -ymin);
        double area = 0., sx = 0., sy = 0., absolute_cross = 0.;
        for (std::size_t k=0; k<polygon.size(); ++k) {
            const auto& p=polygon[k];
            const auto& q=polygon[(k+1)%polygon.size()];
            const double cross=p.x*q.y-p.y*q.x;
            area += cross;
            absolute_cross += std::abs(cross);
            sx += (p.x+q.x)*cross;
            sy += (p.y+q.y)*cross;
        }
        fallback[cell] = 0;
        if (polygon.size()<3 || std::abs(area)<1e-12 || !std::isfinite(area)) {
            // Re-evaluate with the original arithmetic before deciding that a
            // cell is degenerate; do not invent a replacement target.
            fallback[cell] = 1;
            output[2*cell] = output[2*cell+1] = 0.;
        } else {
            output[2*cell] = sx/(3*area);
            output[2*cell+1] = sy/(3*area);
            const double cancellation = absolute_cross/std::abs(area);
            stats[0] = std::max(stats[0], cancellation);
            if (cancellation>1e6 || !std::isfinite(output[2*cell]) || !std::isfinite(output[2*cell+1]))
                fallback[cell] = 2;
        }
    }
    stats[1] = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return 0;
}

