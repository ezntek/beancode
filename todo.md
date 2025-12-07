# Planned Features

## Low-hanging fruit

- [x] `FORMAT` libroutine for printf-like formatting
- [x] Tracer
  - [x] Refactoring HTML to be more modular
  - [x] Dark/light mode CSS, theming?
  - [x] Common class names for bcweb (beancode-web)

## High Priority

- [x] Lexer Refactor
  - [x] Destringify tokens
- [x] Library routine refactor
  - [x] Proper FFI interface
  - [x] Variadic arguments
  - [x] Array passing support
- [x] MAKEARRAY/CLEARARRAY whatever library routine (INITARRAY)
- [x] BCValue shrink (Proper tagged union)
- [x] AST Optimizer
  - [x] static expression evaluation (includes library routines)
  - [x] replacing constants
  - [x] inlining library routines
  - [x] insert native Python calls when possible (maybe?) 

## Low Priority

(0.7)

- [ ] Formatter
- [ ] Make string concatenation faster

(0.8)

- [ ] Multiple Error Reporting 
- [ ] Refactor error implementation (no more hardcoded file names with file IDs, proper context support)
- [ ] Python Compiler? (Turning an AST into an exec call)
- [ ] Static/Semantic Analyzer
- [ ] JS/Lua/C transpiler
- [ ] Proper FFI
