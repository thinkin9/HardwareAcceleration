package chipyard

import chisel3._

import org.chipsalliance.cde.config.{Config}

// ---------------------
// Ibex Configs
// ---------------------

// Multi-core and 32b heterogeneous configs are supported

class IbexConfig extends Config(
  new ibex.WithNIbexCores(1) ++
  new chipyard.config.AbstractConfig)
