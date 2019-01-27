use clap::{App, Arg};
use gnuplot;
use gnuplot::{AxesCommon, Figure, PlotOption};

use std::f64;
use std::fs;
use std::io;
use std::vec::Vec;

#[derive(Clone)]
struct State {
    t: f64,
    x: f64,
    v: f64,
    a: f64,
}

struct CollectedData {
    step: f64,
    x: Vec<f64>,
    v: Vec<f64>,
    a: Vec<f64>,
}

impl CollectedData {
    fn new(rounds: usize, step: f64) -> CollectedData {
        return CollectedData {
            step: step,
            x: Vec::with_capacity(rounds),
            v: Vec::with_capacity(rounds),
            a: Vec::with_capacity(rounds),
        };
    }
    fn add(&mut self, s: &State) {
        self.x.push(s.x);
        self.v.push(s.v);
        self.a.push(s.a);
    }
    fn print_phase_tsv<W: io::Write>(&self, w: &mut W) {
        let mut t: f64 = 0.;
        for i in 0..self.x.len() {
            write!(
                w,
                "{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
                t, self.x[i], self.v[i], self.a[i]
            )
            .unwrap();
            t += self.step;
        }
    }
}

trait SolverIterator {
    fn next(&self, s: State, step: f64) -> State;
    fn initial(&self) -> State;
}

// Van der Pol oscillator.
struct VDP {
    x0: f64,
    v0: f64,
    omega0: f64,
    a: f64,
    eta: f64,
}

impl SolverIterator for VDP {
    fn initial(&self) -> State {
        return State {
            t: 0.,
            x: self.x0,
            v: self.v0,
            a: -self.eta * (self.x0 * self.x0 - self.a * self.a) * self.v0
                - self.omega0 * self.omega0 * self.x0,
        };
    }
    fn next(&self, s: State, step: f64) -> State {
        let x_next = s.x + s.v * step + 0.5 * s.a * step * step;
        let v_next = s.v + s.a * step;
        let a_next = -self.eta * (x_next * x_next - self.a * self.a) * v_next
            - self.omega0 * self.omega0 * x_next;
        return State {
            t: s.t + step,
            x: x_next,
            v: v_next,
            a: a_next,
        };
    }
}

// Van der Pol oscillator in standard form d^2/dt^2 x + epsilon (1-x^2) d/dt x + x = 0.
struct NormalizedVDP {
    x0: f64,
    v0: f64,
    epsilon: f64,
}

impl SolverIterator for NormalizedVDP {
    fn initial(&self) -> State {
        return State {
            t: 0.,
            x: self.x0,
            v: self.v0,
            a: self.epsilon * (1. - self.x0 * self.x0) * self.v0 - self.x0,
        };
    }
    fn next(&self, s: State, step: f64) -> State {
        let x_next = s.x + s.v * step + 0.5 * s.a * step * step;
        let v_next = s.v + s.a * step;
        let a_next = self.epsilon * (1. - x_next * x_next) * v_next - x_next;
        return State {
            t: s.t + step,
            x: x_next,
            v: v_next,
            a: a_next,
        };
    }
}

// Chaotic mathematical pendulum:
// d^2/dt^2 phi + 2 gamma d/dt phi + g/R sin(phi) = h0 cos(omega * t)
struct ChaoticDrivenPendulum {
    phi0: f64,
    vphi0: f64,
    r: f64,
    g: f64,
    h0: f64,
    gamma: f64,
    omega: f64,
}

impl SolverIterator for ChaoticDrivenPendulum {
    fn initial(&self) -> State {
        return State {
            t: 0.,
            x: self.phi0,
            v: self.vphi0,
            a: self.h0 * 1. - 2. * self.gamma * self.vphi0 - self.g / self.r * self.phi0.sin(),
        };
    }
    fn next(&self, s: State, step: f64) -> State {
        let x_next = s.x + s.v * step + 0.5 * s.a * step * step;
        let v_next = s.v + s.a * step;
        let a_next = self.h0 * (self.omega * s.t).cos()
            - 2. * self.gamma * v_next
            - self.g / self.r * x_next.sin();
        return State {
            t: s.t + step,
            x: x_next,
            v: v_next,
            a: a_next,
        };
    }
}

fn drive<S: SolverIterator>(it: &S, step: f64, rounds: usize) -> CollectedData {
    let mut collected = CollectedData::new(rounds, step);
    let mut current = it.initial();
    for _ in 1..rounds {
        collected.add(&current);
        current = it.next(current, step);
    }
    return collected;
}

// Driver for rendering data using gnuplot.
fn render_phasespace(cd: &CollectedData, dst: &str) -> io::Result<()> {
    let mut figure = Figure::new();
    let axesopts = [
        PlotOption::LineStyle(gnuplot::DashType::Solid),
        PlotOption::LineWidth(1.),
    ];
    figure.set_terminal("pngcairo size 1920, 1080", dst);
    figure
        .axes2d()
        .set_x_axis(true, &axesopts)
        .set_y_axis(true, &axesopts)
        .lines(
            cd.x.iter(),
            cd.v.iter(),
            &[
                PlotOption::LineStyle(gnuplot::DashType::Solid),
                PlotOption::LineWidth(2.),
                PlotOption::Color("red"),
            ],
        )
        .set_x_grid(true)
        .set_x_label("X / Phi", &[])
        .set_y_grid(true)
        .set_y_label("V / d/dtPhi", &[])
        .set_grid_options(
            false,
            &[
                PlotOption::LineStyle(gnuplot::DashType::Dash),
                PlotOption::LineWidth(1.),
                PlotOption::Color("gray"),
            ],
        );
    figure.show();

    Ok(())
}

fn render_xvt(cd: &CollectedData, dst: &str) -> io::Result<()> {
    let mut figure = Figure::new();
    let axesopts = [
        PlotOption::LineStyle(gnuplot::DashType::Solid),
        PlotOption::LineWidth(1.),
    ];

    figure.set_terminal("pngcairo size 1920, 1080", dst);
    figure
        .axes2d()
        .set_x_axis(true, &axesopts)
        .set_y_axis(true, &axesopts)
        .lines(
            (0..cd.x.len()).map(|t| t as f64 * cd.step),
            cd.x.iter(),
            &[
                PlotOption::Caption("x(t)"),
                PlotOption::LineStyle(gnuplot::DashType::Solid),
                PlotOption::LineWidth(2.),
                PlotOption::Color("red"),
            ],
        )
        .lines(
            (0..cd.x.len()).map(|t| t as f64 * cd.step),
            cd.v.iter(),
            &[
                PlotOption::Caption("v(t)"),
                PlotOption::LineStyle(gnuplot::DashType::Solid),
                PlotOption::LineWidth(2.),
                PlotOption::Color("blue"),
            ],
        )
        .set_x_grid(true)
        .set_x_label("X / Phi", &[])
        .set_y_grid(true)
        .set_y_label("V / d/dtPhi", &[])
        .set_grid_options(
            false,
            &[
                PlotOption::LineStyle(gnuplot::DashType::Dash),
                PlotOption::LineWidth(1.),
                PlotOption::Color("gray"),
            ],
        )
        .set_legend(
            gnuplot::Coordinate::Graph(0.8),
            gnuplot::Coordinate::Graph(0.8),
            &[],
            &[],
        );
    figure.show();
    Ok(())
}

fn getarg<T: std::str::FromStr + std::string::ToString>(
    m: &clap::ArgMatches,
    name: &str,
    default: T,
) -> T {
    m.value_of(name)
        .unwrap_or(&default.to_string())
        .parse()
        .unwrap_or(default)
}

fn main() {
    let matches = App::new("oscillator")
        .version("0.1")
        .about("numerical DE solver")
        .arg(
            Arg::with_name("type")
                .long("type")
                .takes_value(true)
                .help("vdp (van-der-pol, with x0 v0 eta a omega0) or nvdp (normalized van-der-pol, with epsilon x0 v0) or chaoticpendulum (with x0 = phi0, v0 = d/dt phi, r, g, h0 = F/m, gamma, omega)"),
        )
        .arg(
            Arg::with_name("step")
                .long("step")
                .takes_value(true)
                .help("time step in seconds"),
        )
        .arg(
            Arg::with_name("time")
                .long("time")
                .takes_value(true)
                .help("time span to calculate"),
        )
        .arg(
            Arg::with_name("out-phase")
                .long("out-phase")
                .takes_value(true)
                .help("output file for phase space"),
        )
        .arg(
            Arg::with_name("out-xvt")
                .long("out-xvt")
                .takes_value(true)
                .help("output file for x-v-t diagram"),
        )
        .arg(Arg::with_name("a").long("a").takes_value(true))
        .arg(Arg::with_name("epsilon").long("epsilon").takes_value(true))
        .arg(Arg::with_name("eta").long("eta").takes_value(true))
        .arg(Arg::with_name("g").long("g").takes_value(true))
        .arg(Arg::with_name("gamma").long("gamma").takes_value(true))
        .arg(Arg::with_name("h0").long("h0").takes_value(true))
        .arg(Arg::with_name("omega").long("omega").takes_value(true))
        .arg(Arg::with_name("omega0").long("omega0").takes_value(true))
        .arg(Arg::with_name("r").long("r").takes_value(true))
        .arg(Arg::with_name("v0").long("v0").takes_value(true))
        .arg(Arg::with_name("x0").long("x0").takes_value(true))
        .get_matches();

    let typ = getarg(&matches, "type", "vdp".to_string());
    let time = getarg(&matches, "time", 0.5);
    let step = getarg(&matches, "step", 0.001);
    let rounds = (time / step) as usize;
    let data;

    if typ == "vdp" {
        let vdp = VDP {
            x0: getarg(&matches, "x0", 1.),
            v0: getarg(&matches, "v0", 0.),
            eta: getarg(&matches, "eta", 1.4),
            a: getarg(&matches, "a", 3.),
            omega0: getarg(&matches, "omega0", 20.),
        };
        data = drive(&vdp, step, rounds);
    } else if typ == "nvdp" {
        let nvdp = NormalizedVDP {
            x0: getarg(&matches, "x0", 1.),
            v0: getarg(&matches, "v0", 0.),
            epsilon: getarg(&matches, "epsilon", 0.),
        };
        data = drive(&nvdp, step, rounds);
    } else if typ == "chaoticpendulum" {
        let cdp = ChaoticDrivenPendulum {
            phi0: getarg(&matches, "x0", 0.1),
            vphi0: getarg(&matches, "v0", 0.),
            r: getarg(&matches, "r", 1.),
            g: getarg(&matches, "g", 9.81),
            h0: getarg(&matches, "h0", 1.0),
            gamma: getarg(&matches, "gamma", 0.1),
            omega: getarg(&matches, "omega", 1.),
        };
        data = drive(&cdp, step, rounds);
    } else {
        unimplemented!()
    }

    let out = getarg(&matches, "out-phase", "".to_string());
    let out_xvt = getarg(&matches, "out-xvt", "".to_string());

    if !out_xvt.is_empty() && out_xvt.ends_with("png") {
        render_xvt(&data, &out_xvt).unwrap();
    }

    if !out.is_empty() && out.ends_with("png") {
        render_phasespace(&data, &out).unwrap();
    } else if !out.is_empty() && out.ends_with("dat") {
        let mut file = fs::OpenOptions::new()
            .truncate(true)
            .write(true)
            .create(true)
            .open(out)
            .unwrap();
        data.print_phase_tsv(&mut file);
    }

    println!("Specify --out-phase to render the phase space. Supported formats are .dat (TSV) and .png. The .dat file contains all coordinates.");
    println!("Specify --out-xvt to render a x/v diagram. Supported format is .png");
}
