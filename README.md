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

lowercase keywords are supported. you may also pass arrays into functions. you may also declare functions inside functions.

## installation

why

## running

`python main.py yourfile.bean` (the extension does not matter)

## quirks

* ***the errors are complete unintelligible dogfeces***. I will not fix them. I do not want to fix them.
* Lowercase keywords are supported.
* ***TODO: allow arrays of unknown size to be parsed and passed into functions/procs***
* ***TODO: case of***
* ***TODO: else if if i feel freaky***
* ***a fecesload of testing***
* the code is horrible (unidiomatic. i miss C. i miss tagged unions from rust. i _strongly dislike_ oop.)
