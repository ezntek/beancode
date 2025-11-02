# beancode

## WARNING

This is the **development branch of beancode**. If you want stable builds or specifically versioned releases, head to the respective branches. This branch may have breaking changes and bugs added/removed anytime.

---

This is a tree-walking interpreter for IGCSE pseudocode, as shown in the [2023-2025 syllabus](https://ezntek.com/doc/2023_2025_cs_syllabus.pdf), written in Python (3.10+).

***IMPORTANT:*** Some examples using [raylib](https://github.com/raysan5/raylib) are provided. They were written entirely for fun; in order to run those examples one must install the `raylib` package for those examples to run, else, you will get an error.

***IMPORTANT:*** I do not guarantee this software to be bug-free; most major bugs have been patched by now, and the interpreter has been tested against various examples and IGCSE Markschemes. Version 0.3.0 and up should be relatively stable, but if you find bugs, please report them and I will fix them promptly. **consider this software (all `0.x` versions) unstable and alpha-quality, breaking changes may happen at any time.**

Once I deem it stable enough, I will tag `v1.0.0`.

## Dependencies

None!

## Installation

### Notice

If you want to enjoy actually good performance, ***please use PyPy!*** It is a Python JIT (Just-in-time) compiler, making it far faster than the usual Python implementation CPython. I would recommend you use PyPy even if you werent using this project for running serious work, but it works really well for this project.

Check the appendix for some stats.

### Installing from PyPI (pip)

* `pip install --break-system-packages beancode` ***since this package does not actually have dependencies, you can pass `--break-system-packages` safely. It can still be a bad idea.***
* `pipx install beancode` (the safer way, but you need `pipx` on your system first.)

### Installing from this repository

* Clone the respository with `git clone https://github.com/ezntek/beancode`
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

## The REPL

The REPL (or Read-Eval-Print-Loop) allows you to write beancode directly in your terminal. Run beancode (with the above instructions) without any arguments (i.e. just the command), and you will be dropped into this prompt:

```
=== welcome to beancode 0.5.0 ===
Using Python 3.13.7 (main, Sep  9 2025, 16:20:24) [GCC 15.2.1 20250813]
type ".help" for a list of REPL commands, ".exit" to exit, or start typing some code.
>> 
```

You can immediately begin typing Pseudocode, and all errors will be reported to you. If you want to run a beancode script, you can just `INCLUDE "MyScript.bean"` to execute it, and then immediately return to the REPL.

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

## Extra Features

There are many extra features, or beancode extensions, which are not standard to IGCSE Pseudocode.

*In theory, you can write this in your exams, and examiners should understand what you are doing, but it is safer to steer away from these extensions for formal purposes, and only use them for your own personal testing.*

1. **Lowercase keywords are supported; but cases may not be mixed. All library routines are fully case-insensitive.**
2. Includes can be done with `INCLUDE "file.bean"`, relative to the file.
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
   And many more. Look at the full list of library routines by running `help("libroutines")` in the REPL.
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
7. Get the type of any value as a string with `TYPE(value)` or `TYPEOF(value)`.
8. Array literals are supported:
 * `Arr <- {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}`
   It will declare an array of the given type, with the size of the elements in it. You may not append items,
   but you can always get the type of the array with `TYPEOF(Arr)`.
9. You can directly assign variables without declaring its type through type inference:
   ```
   X <- 5
   OUTPUT X // works
   ```
10. If you need more help, or a reference to some features in the language, always check out the `help()` library routine. Pass in a string, like `help("help")` to get help.
   

### Tips and Tricks

1. If you want to pass arrays of an unknown size into functions or procedures, you can read the `BubbleSort.bean` example in the examples directory. In short, you can pass both the array and the length of it together:

  ```
  PROCEDURE PrintArray(Data: ARRAY[1:End] OF INTEGER, End: INTEGER)
      FOR Counter <- 1 TO End
          OUTPUT Data[Counter]
      NEXT Counter
  ENDPROCEDURE
  ```
2. If you somehow do want multiple statement in CASE OFs, you can abuse the following trick:
  ```
  CASE OF Var
      CASE 'a': IF TRUE THEN
          OUTPUT "Message 1"
          OUTPUT "Message 2"
      ENDIF
      // Other stuff goes here
  ENDCASE
  ```
  This works because beancode only looks for one statement after your case. Since an if statement is just a statement, this will work just fine :)

3. Don't declare variables when you use the REPL! Just assign them.
4. If you ever need to work with arrays in the REPL, just use array literals! Just like in Python with square brackets `[]`, you can declare and initialize an array already with elements in it on the spot. It works for 2D arrays too!
  ```
  EvenNumbers <- {2, 4, 6, 8, 10}
  // EvenNumbers is now an ARRAY[1:5] OF INTEGER
  Grid <- {
    {1, 2},
    {3, 4},
    {5, 6},
  }
  // Grid is now an ARRAY[1:3,1:2] OF INTEGER.
  // You can leave the declaration on the same line
  ```
5. If you are ever in a situation like so:
  ```
  >> FOR i <- 2 TO 10 STEP 2
  .. OUTPUT "Num: ", i
  ..
  ```
  And you want to quit editing the current block, type `Ctrl-C`, which will drop you back to the normal REPL.
  If you want to properly exit the REPL, type `.exit` or `.quit`.

6. If you are stuck in an infinite loop (that you may have written), you can always hit `Ctrl-C` to terminate the running program or the REPL.

## quirks

* ***Multiple statements in CASE OFs are not supported! Therefore, the following code is illegal:***
  ```
  CASE OF Var
      'a': OUTPUT "foo"
           OUTPUT "bar"
  ENDCASE
  ```
  Please put your code into a procedure instead, or use the recommended trick above.
* No-declare assignments are only bound to the `local block-level scope`, they are not global. Please declare it globally if you want to use it like a global variable.
* ***File IO is completely unsupported.*** You might get cryptic errors if you try.
* Not more than 1 parse error can be reported at one time.
* Variable shadowing is ***extremely weird!***
  * You can shadow variables perfectly fine in functions and procedures, check `examples/ShadowingDemo.bean` for examples. However, you cannot do so in
    any other scope, as there is no concept of a local/global variable, and the interpreter does not know when you declared a variable.
  * You can shadow variable declarations of the same type and same kind (i.e. variable or constant), but you cannot, lets say, shadow a variable with a constant, etc.

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

It's really bad. However, PyPy makes it a lot better. Here's some data for the PrimeTorture benchmark in the examples, ran on an i7-14700KF with 32GB RAM on Arch Linux:

|Language|Time Taken (s)|
|--------|----------|
|beancode (CPython 3.13.5)|148|
|beancode (PyPy3 7.3.20)|11|
|beancode (CPython Nuitka)|185|
|Python (CPython 3.13.5)|0.88|
|Python (PyPy3)|0.19|
|C (gcc 15.2.1)|0.1|

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
