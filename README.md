# beancode

## information

***if you want the working python version, head over to the `py` branch. this is the cursed zig version.***

## standard

The language is similar to IGCSE pseudocode, with some differences to make it more bearable to write. If you want a fully-conformant IGCSE pseudocode interpreter with extra features like FFI, check out the `py` branch. I call this `Beancode2`

Modifications include:
* lowercase keywords are preferred.
* array literals are supported with `'{' expr {, expr} '}'` like `{ 2, 3, 4 }`
* `declare` -> `var`
* `constant` -> `const`
* `output` -> `print`
* `input` -> `read`
* `endif/endwhile/next X/endcase` -> `end`
* `case of` -> `switch`
  * syntax looks as follows:
    ```
    switch <expr>
        case "fallthrough" continue
        case "apple"
            print "apples are very krunchy"
        end
        case "kumquat"
            print "don't eat kumquats like candy!"
        continue
        case "citrus"
            print "citrus is yummy!"
        end
    end
    ```
* `else if` supported
* both repeat-until and do-while are supported
* `call` byebye (and no more procedures)
* `<>` -> `!=`
* `=` -> `==`
* `<-` -> `=`
* `integer` -> `int`
* `real` -> `float`
* `boolean` -> `bool`
* functions are declared as
  ```
  function xyz(): int
      return 5
  end
  ```
  the arg list must also exist regardless of if there are arguments or not

## installation

why
