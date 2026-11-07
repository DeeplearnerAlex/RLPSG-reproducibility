// Exhaustive pair/subarc extrema without a Python np.roots loop.
// Derivative critical points partition [0,h] into monotone cubic intervals;
// evaluate every partition point and bisect every strict sign-crossing root.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static double squared_distance(double rx, double ry, double ux, double uy,
                               double bx, double by, double t) {
    const double x=rx+t*ux+t*t*bx, y=ry+t*uy+t*t*by;
    return x*x+y*y;
}

static double cubic(double c3, double c2, double c1, double c0, double s) {
    return ((c3*s+c2)*s+c1)*s+c0;
}

static double swept_squared(double rx, double ry, double ux, double uy,
                            double bx, double by, double h, std::int64_t& roots_checked) {
    double best=std::min(rx*rx+ry*ry,squared_distance(rx,ry,ux,uy,bx,by,h));
    const double c0=rx*ux+ry*uy;
    const double c1=ux*ux+uy*uy+2*(rx*bx+ry*by);
    const double c2=3*(ux*bx+uy*by);
    const double c3=2*(bx*bx+by*by);
    const double variation=std::abs(c1)*h+std::abs(c2)*h*h+std::abs(c3)*h*h*h;
    if (std::abs(c0)>variation+1e-20) return best;

    // Normalize time to [0,1] and coefficients to avoid needless scale loss.
    double d3=c3*h*h*h,d2=c2*h*h,d1=c1*h,d0=c0;
    const double scale=std::max({std::abs(d3),std::abs(d2),std::abs(d1),std::abs(d0)});
    if (scale==0) return best;
    d3/=scale; d2/=scale; d1/=scale; d0/=scale;
    double split[4]={0.,1.,0.,0.};
    int count=2;
    const double A=3*d3,B=2*d2,C=d1;
    auto add_split=[&](double s) {
        if (std::isfinite(s) && s>0. && s<1. && count<4) split[count++]=s;
    };
    if (A==0.) {
        if (B!=0.) add_split(-C/B);
    } else {
        const double discriminant=B*B-4*A*C;
        if (discriminant>=0.) {
            const double radical=std::sqrt(discriminant);
            const double stable=-.5*(B+std::copysign(radical,B));
            if (stable!=0.) {
                add_split(stable/A);
                add_split(C/stable);
            } else {
                add_split(-B/(2*A));
            }
        }
    }
    std::sort(split,split+count);
    for (int k=0;k<count;++k) {
        best=std::min(best,squared_distance(rx,ry,ux,uy,bx,by,split[k]*h));
    }
    for (int k=0;k<count-1;++k) {
        double lo=split[k],hi=split[k+1];
        double flo=cubic(d3,d2,d1,d0,lo),fhi=cubic(d3,d2,d1,d0,hi);
        if (flo==0. || fhi==0. || !(std::signbit(flo)!=std::signbit(fhi))) continue;
        for (int iteration=0;iteration<60;++iteration) {
            const double mid=.5*(lo+hi);
            if (mid==lo || mid==hi) break;
            const double value=cubic(d3,d2,d1,d0,mid);
            if (value==0.) { lo=hi=mid; break; }
            if (std::signbit(value)==std::signbit(flo)) { lo=mid;flo=value; }
            else { hi=mid;fhi=value; }
        }
        ++roots_checked;
        best=std::min(best,squared_distance(rx,ry,ux,uy,bx,by,.5*(lo+hi)*h));
    }
    return best;
}

extern "C" EXPORT int accelerated_arc_metrics(
    std::int64_t arcs,std::int64_t robots,std::int64_t pair_count,
    const double* position,const double* velocity,const double* acceleration,
    const double* duration,const std::int64_t* pairs,
    double contact_distance,double radius,double domain_side,
    double* output,std::uint8_t* contact_flags) {
    const auto start=std::chrono::steady_clock::now();
    std::fill(contact_flags,contact_flags+pair_count,0);
    double min_squared=std::numeric_limits<double>::infinity();
    double min_wall=std::numeric_limits<double>::infinity();
    double max_speed=0.,max_acceleration=0.,travel=0.;
    std::int64_t roots_checked=0;
    const double nodes[3]={.5-std::sqrt(15.)/10.,.5,.5+std::sqrt(15.)/10.};
    const double weights[3]={5./18.,4./9.,5./18.};
    const double threshold=std::max(0.,contact_distance-1e-8);
    for (std::int64_t step=0;step<arcs;++step) {
        const double h=duration[step];
        const std::int64_t offset=step*robots*2;
        for (std::int64_t i=0;i<robots;++i) {
            const std::int64_t index=offset+i*2;
            const double vx=velocity[index],vy=velocity[index+1];
            const double ax=acceleration[index],ay=acceleration[index+1];
            max_speed=std::max({max_speed,std::hypot(vx,vy),std::hypot(vx+h*ax,vy+h*ay)});
            max_acceleration=std::max(max_acceleration,std::hypot(ax,ay));
            for (int coordinate=0;coordinate<2;++coordinate) {
                const double p=position[index+coordinate],v=velocity[index+coordinate],a=acceleration[index+coordinate];
                const double end=p+h*v+.5*h*h*a;
                double low=std::min(p,end),high=std::max(p,end);
                if (std::abs(a)>1e-30) {
                    const double turning=-v/a;
                    if (turning>0. && turning<h) {
                        const double extreme=p+turning*v+.5*turning*turning*a;
                        low=std::min(low,extreme); high=std::max(high,extreme);
                    }
                }
                min_wall=std::min({min_wall,low-radius,domain_side-radius-high});
            }
            for (int node=0;node<3;++node) {
                const double t=nodes[node]*h;
                travel+=weights[node]*h*std::hypot(vx+t*ax,vy+t*ay);
            }
        }
        for (std::int64_t pair=0;pair<pair_count;++pair) {
            const std::int64_t i=offset+2*pairs[2*pair],j=offset+2*pairs[2*pair+1];
            const double value=swept_squared(position[i]-position[j],position[i+1]-position[j+1],
                            velocity[i]-velocity[j],velocity[i+1]-velocity[j+1],
                            .5*(acceleration[i]-acceleration[j]),.5*(acceleration[i+1]-acceleration[j+1]),
                            h,roots_checked);
            min_squared=std::min(min_squared,value);
            if (value<threshold*threshold) contact_flags[pair]=1;
        }
    }
    output[0]=std::sqrt(min_squared);
    output[1]=min_wall;
    output[2]=max_acceleration;
    output[3]=max_speed;
    output[4]=travel;
    output[5]=static_cast<double>(roots_checked);
    output[6]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return 0;
}

