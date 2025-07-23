# beancode

***WARNING***: This is *NOT* my best work. please do *NOT* assume my programming ability to be this, and do *NOT* use this project as a reference for yours. The layout is horrible. The code style is horrible. The code is not idiomatic. I went through 607,587,384 hacks and counting just for this project to work.

## information

This is a very cursed non-optimizing super-cursed super-cursed-pro-max-plus-ultra IGCSE pseudocode tree-walk interpreter written in the best language, Python.

(I definitely do not have 30,000 C projects and I definitely do not advocate for C and the burning of Python at the stake for projects such as this).

It's slow, it's horrible, it's hacky, but it works :) and if it ain't broke, don't fix it.

This is my foray into compiler engineering; through this project I have finally learned how to perform recursive-descent parsing. I will most likely adapt this into C/Rust (maybe not C++) and play with a bytecode VM sooner or later (with a different language, because Python is slow and does not have null safety in 2025).

`</rant>`

## standard

this is a tree-walker for IGCSE pseudocode, as shown in the [2023-2025 syllabus](https://ezntek.com/doc/2023_2025_cs_syllabus.pdf) (pseudocode was this way for literal millenia).

## installation

why

## running

note: the extension does not matter

`python -m beancode --file yourfile.bean`
`python main.py --file yourfile.bean` 


## xtra feachurse™

there are many extra features, which are not standard to IGCSE Pseudocode.

1. Lowercase keywords are supported; but cases may not be mixed. All library routines are fully case-insensitive.
2. Includes can be done with `include "file.bean"`, relative to the file.
 * Mark a declaration, constant, procedure, or function as exportable with `EXPORT`, like `EXPORT DECLARE X:INTEGER`.
 * Symbols marked as export will be present in whichever scope the include was called.
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

## quirks

* ***the errors are complete unintelligible dogfeces***. I will not fix them. I do not want to fix them.
* Lowercase keywords are supported.
* ***TODO: allow arrays of unknown size to be parsed and passed into functions/procs***
* ***TODO: else if if i feel freaky***
* ***a fecesload of testing***
* the code is horrible (unidiomatic. i miss C. i miss tagged unions from rust. i _strongly dislike_ oop.)
