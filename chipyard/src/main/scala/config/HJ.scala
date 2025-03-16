// HeteroConfigs.scala GemminiSoCConfigs.scala
package chipyard
import org.chipsalliance.cde.config.{Config}
import freechips.rocketchip.diplomacy.{AsynchronousCrossing}

// GemminiCustomConfigs.scala
package gemmini
import org.chipsalliance.cde.config.{Parameters}  // import org.chipsalliance.cde.config.{Config, Parameters}
import chisel3._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey
import freechips.rocketchip.tile.BuildRoCC

/*
Custom Gemmini Configuration
Custom Gemmini SoC Configuration
Custom HW Configuration
*/


/*
Custom Gemmini Functional Configuration
1. baselineInferenceConfig  @ baseline configuration in Gemmini paper
2. transformerInferenceConfig  @ hw configuration in Full stack optimization ~~ paper
*/
object HJGemminiCustomConfigs {
  // Default configurations
  val defaultConfig = GemminiConfigs.defaultConfig
  
  // 1. baselineInferenceConfig
  val baselineInferenceConfig = defaultConfig.copy(
    has_training_convs = false

    dataflow = Dataflow.WS
  )
  // 2. transformerInferenceConfig
  val transformerInferenceConfig = defaultConfig.copy(
    meshRows = 16,
    meshColumns = 16,

    has_training_convs = false,
    has_max_pool = false,
    has_normalizations = true,

    sp_capacity = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(256),

    dataflow = Dataflow.WS
  )

  val customConfig = baselineInferenceConfig
}


/*
Custom Gemmini Configuration
*/
class HJGemminiCustomConfig[T <: Data : Arithmetic, U <: Data, V <: Data](
  gemminiConfig: GemminiArrayConfig[T,U,V] = HJGemminiCustomConfigs.customConfig
) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val gemmini = LazyModule(new Gemmini(gemminiConfig))
      gemmini
    }
  )
})


/*
Custom HW Configuration
1. Rocket   1 core
2. BOOM     1 core
3. Rocket   1 core  Gemmini
4. BOOM     1 core  Gemmini
*/
// 1. Rocket   1 core
class HJRocketConfig extends Config(
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.AbstractConfig
)

// 2. BOOM     1 core
class HJLargeBoomConfig extends Config(
  new boom.common.WithNLargeBooms(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)

// 3. Rocket   1 core  Gemmini
class HJRocketGemminiConfig extends Config(
  new gemmini.HJGemminiCustomConfig ++
  new freechips.rocketchip.subsystem.WithNBigCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)

// 4. BOOM     1 core  Gemmini
class HJBoomGemminiConfig extends Config(
  new gemmini.HJGemminiCustomConfig ++
  new boom.common.WithNLargeBooms(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
