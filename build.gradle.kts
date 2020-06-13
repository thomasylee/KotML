import org.gradle.api.tasks.testing.logging.TestExceptionFormat
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    // Apply the Kotlin JVM plugin to add support for Kotlin.
    id("org.jetbrains.kotlin.jvm") version "1.3.72"

    // Apply the java-library plugin for API and implementation separation.
    `java-library`

    // Use ktlint for Kotlin style enforcement.
    id("org.jlleitschuh.gradle.ktlint") version "9.2.1"

    // Dokka for KDoc documentation generation.
    id("org.jetbrains.dokka") version "0.10.1"
}

repositories {
    jcenter()
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

tasks {
    test {
        useJUnitPlatform()

        testLogging {
            // Comment this line to hide stdout in test output.
            showStandardStreams = true

            exceptionFormat = TestExceptionFormat.FULL
        }
    }

    dokka {
        outputFormat = "html"
        outputDirectory = "$buildDir/dokka"
    }
}

dependencies {
    // Align versions of all Kotlin components
    implementation(platform("org.jetbrains.kotlin:kotlin-bom"))

    // Use the Kotlin JDK 8 standard library.
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")

    // OpenBLAS for faster matrix operations.
    implementation("org.bytedeco:openblas-platform:0.3.9-1.5.3")
    implementation("org.bytedeco:mkl-platform-redist:2020.1-1.5.3")

    // Use the Kotlin test library.
    testImplementation("org.jetbrains.kotlin:kotlin-test")

    // Use the Kotlin JUnit integration.
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit")

    // Use JUnit 5.
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.6.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.6.2")
}
