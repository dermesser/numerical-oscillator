use clap::{App, Arg};

use plotlib;
use plotlib::scatter::{Scatter, Style};
use plotlib::style::Point;
use plotlib::view::View;

use svg;

use std::f64;
use std::fs;
use std::io;
use std::vec::Vec;

#[derive(Clone)]
struct State {
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

fn render_phasespace(cd: CollectedData, dst: &str) -> io::Result<()> {
    let (xmin, xmax) = cd.x.iter().fold((f64::MAX, f64::MIN), |(min, max), x| {
        (
            if *x < min { *x } else { min },
            if *x > max { *x } else { max },
        )
    });
    let (ymin, ymax) = cd.v.iter().fold((f64::MAX, f64::MIN), |(min, max), x| {
        (
            if *x < min { *x } else { min },
            if *x > max { *x } else { max },
        )
    });
    println!("x: {:?} v: {:?}", (xmin, xmax), (ymin, ymax));

    let x = cd.x.into_iter();
    let v = cd.v.into_iter();
    let zipped: Vec<(f64, f64)> = x.zip(v).collect();
    let scatter = Scatter::from_vec(&zipped).style(Style::new().colour("#FF0000").size(1.));

    let group = View::new()
        .add(&scatter)
        .x_range(xmin, xmax)
        .y_range(ymin, ymax)
        .x_label("X")
        .y_label("V");
    plotlib::page::Page::single(&group).save(dst);
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
                .help("vdp (van-der-pol) or mp (mathematical pendulum)"),
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
        .arg(Arg::with_name("x0").long("x0").takes_value(true))
        .arg(Arg::with_name("v0").long("v0").takes_value(true))
        .arg(Arg::with_name("omega0").long("omega0").takes_value(true))
        .arg(Arg::with_name("eta").long("eta").takes_value(true))
        .arg(Arg::with_name("a").long("a").takes_value(true))
        .get_matches();

    let typ = getarg(&matches, "type", "vdp".to_string());
    let time = getarg(&matches, "time", 0.5);
    let step = getarg(&matches, "step", 0.001);
    let rounds = (time / step) as usize;

    if typ == "vdp" {
        let vdp = VDP {
            x0: getarg(&matches, "x0", 1.),
            v0: getarg(&matches, "v0", 0.),
            eta: getarg(&matches, "eta", 1.4),
            a: getarg(&matches, "a", 3.),
            omega0: getarg(&matches, "omega0", 20.),
        };
        let data = drive(&vdp, step, rounds);
        let out = getarg(&matches, "out-phase", "".to_string());
        if !out.is_empty() && out.ends_with("svg") {
            render_phasespace(data, &out).unwrap();
        } else if !out.is_empty() && out.ends_with("dat") {
            let mut file = fs::OpenOptions::new()
                .truncate(true)
                .write(true)
                .create(true)
                .open(out)
                .unwrap();
            data.print_phase_tsv(&mut file);
        } else {
            println!("Specify --out-phase to render the phase space");
        }
    } else {
        unimplemented!()
    }
}
