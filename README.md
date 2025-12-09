# beancode

This is a fully syllabus-compliant (optimizing) interpreter for IGCSE pseudocode, as shown in the [2023-2025 syllabus](https://ezntek.com/doc/2023_2025_cs_syllabus.pdf) and is compatible with all later versions. It is written in Python, and is compatible with all versions above and including version 3.10.

This project aims to be as portable as possible, and therefore has **no** external dependencies.

***IMPORTANT:*** I do not guarantee this software to be bug-free; most major bugs have been patched by now, and the interpreter has been tested against various examples and IGCSE Markschemes. Version 0.3.0 and up should be relatively stable, but if you find bugs, please report them and I will fix them promptly. **consider this software (all `0.x` versions) unstable and alpha-quality, breaking changes may happen at any time.**

Once I deem it stable enough, I will tag `v1.0.0`.

## Installation

If you want to enjoy actually good performance, ***please use PyPy!*** It is a [Python JIT (Just-in-time) compiler](https://pypy.org), making it far faster than the usual Python implementation CPython. I would recommend you use PyPy even if you werent using this project for running serious work, but it works really well for this project.

Check the appendix for some stats.

### Installing from PyPI (pip)

* `pip install --break-system-packages beancode` ***Since this package does not actually have dependencies, you can pass `--break-system-packages` safely. Your system will not in fact break.***
* `pipx install beancode` (this is the safer way, but you need `pipx` on your system first.)

To upgrade:

* `pip install --break-system-packages --force --upgrade beancode`
* `pipx install --force beancode`

### Installing from this repository

* Clone the respository with `git clone https://github.com/ezntek/beancode`
* `cd beancode`
* `pip install . --break-system-packages` OR `pipx install .`

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

You can either choose to run `pip install . --break-system-packages`, which does not actually cause any issues, as to my knowledge, nobody packages beancode outside of PyPI. You can always run it in a virtual environment.

Either way, it is still recommended to use `pipx`, as all the hard work of isolating beancode is done for you.

## Running

*note: the extension of the source file does not matter, but I recommend `.bean`.*

If you installed it globally:

`beancode file.bean`

If you wish to run it in the project directory:

`python -m beancode file.bean`

You may also run

`./main.py file.bean`

if you are in the project directory.

## The REPL

The REPL (or Read-Eval-Print-Loop) allows you to write beancode directly in your terminal. Run beancode (with the above instructions) without any arguments (i.e. just the command), and you will be dropped into this prompt:

```
=== welcome to beancode 0.6.0 ==
Using Python 3.13.7 (main, Sep  9 2025, 16:20:24) [GCC 15.2.1 20250813]
type ".help" for a list of REPL commands, ".exit" to exit, or start typing some code.
>> 
```

You can immediately begin typing Pseudocode, and all errors will be reported to you. If you want to run a beancode script, you can just `INCLUDE "MyScript.bean"` to execute it, and then immediately return to the REPL. You can also type `.runfile myscript.bean`

You can also start typing dot-commands, which do not control the beancode interpreter, but controls the wrapper around it that provides you with REPL functionality. You can see the list of commands with `.help`, and detailed help is listed below:

### REPL features
   
* `.var [name]` gets information regarding an _existing variable_. It prints its name, type, and value.
  Substitute `[name]` for an actual constant or variable variable's name.
* `.vars` prints information regarding _all variables_.
* `.func [name]` gets information regarding *existing functions* ***or procedures***.
  Substitute `[name]` for an actual function or procedure's name.
* `.funcs` prints information regarding _all functions and procedures_.
* `.delete [name]` lets
* Delete a variable if you need to with `.delete [name]`. (Version `0.3.4` and up)
* reset the entire interpreter's state with `.reset`.
  - This effectively clears all variables, functions, constants, procedures, and included symbols.
* `.trace` traces a script. Pass in the name of the script, and optionally, the variables to trace. An example is provided:
 ```
 .trace MyScript.bean Counter Value
 ```
 if `Counter` and `Value` are variables.

Always consult the `.help` menu for more information.

## Performance Improvements since 0.5

Since this release was meant to boost the performance of beancode, I have done some benchmarking. This is due to the technological improvements made, by using more efficient data structures to represent the AST.

**NOTES:**

1. CPy refers to CPython 3.14.0, PyPy refers to PyPy (version 7.3.20) 3.11.13**
2. All values are in seconds.
3. All values are taken on an Intel Core i7-14700KF with 32GB RAM on Arch Linux (CachyOS kernel), exact results may vary.

| **Benchmark**                 | **0.5.3 (CPy)** | **0.6.0 (CPy)** | **Gains (CPy)** | **0.5.3 (PyPy)** | **0.6.0 (PyPy)** | **Gains (PyPy)** |
|-------------------------------|-----------------|-----------------|-----------------|------------------|------------------|------------------|
| BsortTorture 500 nums         | 4.051           | 2.344           | 1.73x           | 1.166            | 0.698            | 1.67x            |
| QsortTorture 1000 nums        | 3.378           | 3.25            | 1.04x           | 1.434            | 1.283            | 1.12x            |
| PrimeTorture 30000 max        | 2.429           | 1.558           | 1.56x           | 0.528            | 0.382            | 1.38x            |
| raylib_random_rects 400 rects | 3.463           | 1.981           | 1.75x           | 1.406            | 0.737            | 1.91x            |

## Errata

This section shares notable bugs that may impact daily use.

* Some errors will report as `invalid statement or expression`, which is expected for this parser design.

### Version-specific

* Before `v0.3.6`, equal expressions will actually result in `<>` being true. For example, `5 = 5` is `TRUE`, but `5 <> 5` is also `TRUE`.
* Before `v0.4.0`, every word that is not a valid keyword is an identifier. Therefore, you could technically assign dollar signs and backslashes.
* Before `v0.4.0`, function names could be strange, like empty quotation marks.
* Before `v0.4.0`, you could shadow dot-commands in the REPL.
* Before `v0.4.0`, arithmetic with INTEGERs and REALs were very inconsistent, especially in type checking. There may be very weird behavior.
* Before `v0.4.0`, function return types were not checked at all, which may result in unexpected behavior. 
* Before `v0.5.0`, assignments were not properly type-checked sometimes. You could not assign array literals to declared arrays.
* Before `v0.5.0`, you could not assign arrays, even of the same length and type to one another.
* Before `v0.5.0`, you could not declare arrays with only one item in it. 
