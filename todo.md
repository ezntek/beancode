# Planned Features

## Low-hanging fruit

- [x] `FORMAT` libroutine for printf-like formatting
- [ ] Tracer
  - [ ] Refactoring HTML to be more modular
  - [ ] Dark/light mode CSS, theming?
  - [ ] Common class names for bcweb (beancode-web)
  - [ ] JSON output?
  - [ ] Proper debugger? (callbacks)

## High Priority

- [ ] Interpreter class refactor
  - [ ] Proper variable scoping in loops (no more dirty hacky clears)
  - [ ] Reduce/eliminate recursion for block evaluation
  - [ ] Reduce member count
- [x] BCValue shrink (Proper tagged union)
- [ ] Static/Semantic Analyzer
- [ ] AST Optimizer
  - [ ] static expression evaluation (includes library routines)
  - [ ] block optimizations
  - [ ] replacing constants
  - [ ] inlining library routines
  - [ ] insert native Python calls when possible 
  - [ ] AST caching to JSON
- [x] Minor refactor sweep

## Medium Priority

- [ ] AST
  - [ ] Formatter
  - [ ] Decompiler
  - [ ] Python Compiler? (Turning an AST into an exec call)
- [ ] Multiple Error Reporting
- [ ] Refactor error implementation (no more hardcoded file names with file IDs, proper context support)

## Low Priority

- [ ] File IO with raw bytes
- [ ] Proper byte strings and buffers
- [ ] JS/Lua/C transpiler
- [ ] Proper FFI


