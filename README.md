# beancode

This is a tree-walking interpreter for IGCSE pseudocode, as shown in the [2023-2025 syllabus](https://ezntek.com/doc/2023_2025_cs_syllabus.pdf), written in Python (3.10+).

***IMPORTANT:*** Some examples using [raylib](https://github.com/raysan5/raylib) are provided. They were written entirely for fun; in order to run those examples one must install the `raylib` package for those examples to run, else, you will get an error.

***IMPORTANT:*** I do not guarantee this software to be bug-free; most major bugs have been patched by now, and the interpreter has been tested against various examples and IGCSE Markschemes. Version 0.3.0 and up should be relatively stable, but if you find bugs, please report them and I will fix them promptly.

Once I deem it stable enough, I will tag `v1.0.0`.

## Dependencies

* `typed-argument-parser`
* `pipx` if you wish to install it system-wide while being safe.

## Installation

### Notice

If you want to enjoy actually good performance, ***please use PyPy!*** It is a Python JIT (Just-in-time) compiler, making it far faster than the usual Python implementation CPython. I would recommend you use PyPy even if you werent using this project for running serious work, but it works really well for this project.

Check the appendix for some stats.

### Installing from PyPI (pip)

* `pip install --break-system-packages beancode` ***since this package does not actually have dependencies, you can pass `--break-system-packages` safely. It can still be a bad idea.***
* `pipx install beancode` (the safer way)

### Installing from this repository

* Clone the respository with `git clone https://github.com/ezntek/beancode --branch=py --depth=1`
* `cd beancode`
* `pipx install .`

### Notes on using `pip`

If you use pip, you may be faced with an error as such:

```
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try 'pacman -S
    python-xyz', where xyz is the package you are trying to
    install.

=== snip ===

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
```

You can either choose to run `pip install . --break-system-packages`, which is not recommended but is likely to work, or you can run it in a virtual environment.

Either way, it is still recommended to use `pipx`, as all the hard work is done for you.

## Running

*note: the extension of the source file does not matter, but I recommend `.bean`.*

If you installed it globally:

`beancode file.bean`

If you wish to run it in the project directory:

`python -m beancode file.bean`

## extra features™

There are many extra features, which are not standard to IGCSE Pseudocode.

1. Lowercase keywords are supported; but cases may not be mixed. All library routines are fully case-insensitive.
2. Includes can be done with `include "file.bean"`, relative to the file.
 * Mark a declaration, constant, procedure, or function as exportable with `EXPORT`, like `EXPORT DECLARE X:INTEGER`.
 * Symbols marked as export will be present in whichever scope the include was called.
 * Use `include_ffi` to include a bundled FFI module. Support for custom external modules will be added later.
   * `beanray` is an incomplete set of raylib bindings that supports some basic examples.
   * `demo_ffimod` is just a demo.
   * `beanstd` will be a standard library to make testing a little easier.
3. You can declare a manual scope with:
   ```
   SCOPE
       OUTPUT "Hallo, Welt."
   ENDSCOPE
   ```

   Exporting form a custom scope also works:

   ```
   SCOPE
       EXPORT CONSTANT Age <- 5
   ENDSCOPE
   OUTPUT Age
   ```
4. There are many custom library routines:
 * `FUNCTION GETCHAR() RETURNS CHAR`
 * `PROCEDURE PUTCHAR(ch: CHAR)`
 * `PROCEDURE EXIT(code: INTEGER)`
5. Type casting is supported:
 * `Any Type -> STRING`
 * `STRING -> INTEGER` (returns `null` on failure)
 * `STRING -> REAL` (returns `null` on failure)
 * `INTEGER -> REAL`
 * `REAL -> INTEGER`
 * `INTEGER -> BOOLEAN` (`0` is false, `1` is true)
 * `BOOLEAN -> INTEGER`
6. Declaration and assignment on the same line is also supported: `DECLARE Num:INTEGER <- 5`
 * You can also declare variables without types and directly assign them: `DECLARE Num <- 5`
7. Array literals are supported:
 * `Arr <- {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}`
8. Get the type of any value as a string with `TYPE(value)` or `TYPEOF(value)`
9. You can directly assign variables without declaring its type through type inference:
   ```
   X <- 5
   OUTPUT X // works
   ```

## quirks

* ***Multiple statements in CASE OFs are not supported! Therefore, the following code is illegal:***
  ```
  CASE OF Var
      CASE 'a': OUTPUT "foo"
                OUTPUT "bar"
  ENDCASE
  ```
  Please put your code into a procedure instead.
* No-declare assignments are only bound to the `local block-level scope`, they are not global. Please declare it globally if you want to use it like a global variable.
* ***File IO is completely unsupported.*** You might get cryptic errors if you try.
* Not more than 1 parse error can be reported at one time.
* Lowercase keywords are supported.

## Appendix

This turned out to be a very cursed non-optimizing super-cursed super-cursed-pro-max-plus-ultra IGCSE pseudocode tree-walk interpreter written in the best language, Python.

(I definitely do not have 30,000 C projects and I definitely do not advocate for C and the burning of Python at the stake for projects such as this).

It's slow, it's horrible, it's hacky, but it works :) and if it ain't broke, don't fix it.

This is my foray into compiler engineering; through this project I have finally learned how to perform recursive-descent parsing. I will most likely adapt this into C/Rust (maybe not C++) and play with a bytecode VM sooner or later (with a different language, because Python is slow and does not have null safety in 2025).

***WARNING***: This is *NOT* my best work. please do *NOT* assume my programming ability to be this, and do *NOT* use this project as a reference for yours. The layout is horrible. The code style is horrible. The code is not idiomatic. I went through 607,587,384 hacks and counting just for this project to work.

`</rant>`

### Why Python?

Originally this interpreter was only written for me to learn compiler engineering (and how to write a recursive-descent parser and ast walker). However, it quickly spiralled into something usable that I wanted other people to use.

Python was perfect due to its dynamism, and the fact that I could abuse it to the max; and it came in super handy when I realized that students who already have a Python toolchain on their system should only need to run a single `pip install` to use my interpreter. It's meant as a learning tool anyway; it's slow as hell.

### Performance

It's really bad. However, PyPy makes it a lot better. Here's some data for the PrimeTorture benchmark I have, ran on an i7-14700KF with 32GB RAM on Arch Linux:

|Language|Time Taken (s)|
|--------|----------|
|beancode (CPython 3.13.5)|148|
|beancode (PyPy3 7.3.20)|11|
|beancode (CPython Nuitka)|185|
|Python (CPython 3.13.5)|0.88|
|Python (PyPy3)|0.19|
|C (gcc 15.2.1)|0.1|

beancode+cpython: 148 seconds
beancode+pypy3:   11 seconds
beancode+nuitka:  185 seconds
cpython:          0.88 seconds
pypy3:            0.19 seconds
C:                0.1 seconds

## Errata

* Some errors will report as `unused expression`, like the following:
```
for i <- 1 to 10
  output i
nex ti
```
* Some errors will report as `invalid statement or expression`, which is expected for this parser design.

