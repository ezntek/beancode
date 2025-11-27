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
- [ ] MAKEARRAY/CLEARARRAY whatever library routine
- [x] BCValue shrink (Proper tagged union)
- [ ] AST Optimizer
  - [ ] static expression evaluation (includes library routines)
  - [ ] block optimizations
  - [ ] replacing constants
  - [ ] inlining library routines
  - [ ] insert native Python calls when possible 
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


