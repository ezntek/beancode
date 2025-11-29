# Planned Features

## Low-hanging fruit

- [x] `FORMAT` libroutine for printf-like formatting
- [ ] Tracer
  - [ ] Refactoring HTML to be more modular
  - [ ] Dark/light mode CSS, theming?
  - [ ] Common class names for bcweb (beancode-web)
  - [ ] JSON output?

## High Priority

- [ ] Library routine refactor
  - [ ] Proper FFI interface
  - [ ] Variadic arguments
  - [ ] Array passing support
- [x] MAKEARRAY/CLEARARRAY whatever library routine
- [x] BCValue shrink (Proper tagged union)
- [x] AST Optimizer
  - [ ] optimize array initialization
  - [x] static expression evaluation (includes library routines)
  - [ ] block optimizations
    - [ ] folding loops with an always false condition
    - [ ] folding loops that will fail after the first iteration
    - [ ] inlining small loops?
  - [x] replacing constants
  - [x] inlining library routines
  - [x] insert native Python calls when possible (maybe?) 
  - [ ] AST caching to JSON
- [ ] Make string concatenation faster

## Medium Priority

- [ ] AST
  - [ ] Formatter
  - [ ] Decompiler
- [ ] Multiple Error Reporting
- [ ] Refactor error implementation (no more hardcoded file names with file IDs, proper context support)

## Low Priority

- [ ] Python Compiler? (Turning an AST into an exec call)
- [ ] Static/Semantic Analyzer
- [ ] File IO with raw bytes
- [ ] Proper byte strings and buffers
- [ ] JS/Lua/C transpiler
- [ ] Proper FFI


