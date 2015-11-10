#rsrt
rsrt is a small, extensible, physically based ray tracer.  It's very simple, although it does provide quite a few handy abstractions.  The purpose of the project is mainly an exercise in learning the Rust language.  As a learning exercise, the effort has not been put into making the program efficient, but rather into making it clear and expressive.

## Source
Check out the source by cloning the repository:

    $ git clone https://github.com/cdbfoster/rsrt.git

## Building
rsrt uses Rust's Cargo to manage it.  Just use `cargo build` to build the project, or `cargo run` to build and run the project.  Append the `--release` flag to either of those to build in "release" mode.

##Usage
Run `rsrt` from the `target/debug` or `target/release` folder, depending on the options used to build the project.  Or just use `cargo run`, mentioned above.

Currently there is no command-line configurability, and rsrt just spits out a .ppm file containing its render of its default scene.  This is likely to change.

## Contact
Questions and comments can be sent to my email, cdbfoster@gmail.com.

© 2015 Chris Foster
