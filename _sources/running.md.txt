# Running NDS

After completing the [installation](installation.md), and after having all the NDS components properly [configured](configuring.md), you are now ready to run them independently as follows:

## Encoder

To start the [encoder](encoder.md) adjust the [configuration](configuring.md#encoder), then run the script:

```
encoder
```

## Electrophysiology Generator

To start the [ephys generator](ephys_generator.md), adjust the [configuration](configuring.md#ephys-generator), then run the script:

```
ephys_generator
```

## Running Both Systems

To execute both systems you can run:

```
encoder & ephys_generator &
```

## Next Steps

In addition to the core functionality of the encoder and generator of NDS, we provide example components for decoders, user interfaces, and visualization to demonstrate how to incorporate NDS in a closed-loop BCI system.
