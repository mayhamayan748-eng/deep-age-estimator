import { MotionConfig } from "motion/react";
import { Header } from "./components/Header";
import { useApiData } from "./hooks/useApiData";
import { About } from "./sections/About";
import { Footer } from "./sections/Footer";
import { Generate } from "./sections/Generate";
import { Hero } from "./sections/Hero";
import { HowItWorks } from "./sections/HowItWorks";
import { Predict } from "./sections/Predict";

export default function App() {
  const { health, info, offline } = useApiData();

  return (
    // reducedMotion="user": transforms are skipped when the OS asks for reduced motion
    <MotionConfig reducedMotion="user">
      <Header health={health} offline={offline} />
      <main>
        <Hero info={info} />
        <Predict health={health} info={info} offline={offline} />
        <Generate health={health} offline={offline} />
        <HowItWorks info={info} />
        <About info={info} />
      </main>
      <Footer />
    </MotionConfig>
  );
}
