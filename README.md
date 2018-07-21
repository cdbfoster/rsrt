`rsrt` is a small, but very extensible ray tracing framework with some useful implementations.  It is useable, and provides a basic
forward path tracer, a sphere, and some simple materials.

To build and render the default scene, use

    $ cargo run --release

To build and view the documentation, use

    $ cargo doc --open

While it is possible to use `rsrt`'s traits and implementations in other projects, it is not intended to be more than its simple example
binary; its modules and functions are marked `pub` so that `rustdoc` picks them up.  If you want to use them elsewhere, you can move what
you need into a separate `lib.rs` file and have `cargo` build you an actual library.

Questions and comments can be sent to my email, cdbfoster@gmail.com.

© 2018 Chris Foster
