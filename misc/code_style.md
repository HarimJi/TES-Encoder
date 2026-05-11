# Coding Style Guide

## General Rules
- Use C++17 standard.
- Resolve all compiler warnings.
- Apply `clang-format` with the project's [.clang-format](.clang-format) file.
- Follow the existing conventions in the relevant codebase.

## Naming
### File
- Use `snake_case`.
- For large source files, split them into multiple files with a common base name.

### Namespace
- Use `snake_case`, with top-level namespace named `tes`.

### Type
- **Classes**: `PascalCase`
- **Structs**: `snake_case` with `_t` suffix
- **Enumerators**: `snake_case` with `_e` suffix
- **Concepts**: `snake_case`

### Function
- **Global functions**: `snake_case`
- **Member functions**: `camelCase`
- **Static member functions**: `PascalCase`

### Variable
- **Local variables**: `snake_case`
- **Member variables**: `snake_case` with trailing underscore

### Macro
- Use `UPPER_CASE`.

## Header Files
- Use `#pragma once` upfront.
- Keep header files concise yet self-contained.
  - Don't expose unnecessary dependencies.
  - Use forward declarations instead of including related headers.
  - Include all other necessary headers, including project headers.
  - Define a function in a header only if it is short and intended to be inlined.
  - Never use `using namespace` in headers.
- Include headers in the following order, separating each non-empty group with one blank line:
  1. Related header
  2. Standard library headers
  3. CUDA headers
  4. Other libraries' headers
  5. Project headers

## Classes
- Ensure each class instance is valid after construction.
- Use a `struct` only for passive data holders; use a `class` for everything else.
- Do not declare `public` member variables in a `class`; provide getters and setters instead.
- Declare variables before methods, and group by access specifier in the order: `public`, `protected`, and `private`.
- Prefer composition over inheritance; use inheritance only for polymorphic interfaces.

## Miscellaneous
- Use RAII for resource management.
- Pass objects by raw `*` or `&` rather than smart pointers, unless you intend to manipulate their lifetime.
- Use `const` whenever appropriate; place `const` to the right (i.e., `Foo const& foo`).
- Declare local variables right before first use.
- Avoid magic numbers; use named `constexpr`s.
- Use `auto` only when it improves clarity or safety.
- Use `auto*` explicitly when the deduced type is a pointer.
- Use range-based for loop whenever possible.
- Use prefix increment/decrement (i.e., use `++i` instead of `i++`).
- Use C++-style casts.
- Never omit braces for single-line statements, except for obvious cases (e.g., early return).
- Favor early returns over nested conditionals.
- Use `goto` only to escape nested loops.
- Use `[[fallthrough]]` in switch statements to express intentional fall-through between case labels.
- Raw pointers should end with either _h or _d to distinguish between host and device.
